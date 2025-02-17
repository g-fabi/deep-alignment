import argparse

from pytorch_lightning import Trainer, seed_everything

from models.multimodal import MultiModalClassifier

from utils.experiment_utils import generate_experiment_id, load_yaml_to_dict
from utils.training_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    
    # Configs
    parser.add_argument('--experiment_config_path', required=True)
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    
    # Data, modalities and models
    #	Expected to pass modalities, models and checkpoints in the corresponding order
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--protocol', default='cross_subject')
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--models', required=True, nargs='+')
    
    # ssl pre-trained 
    parser.add_argument('--ssl_pretrained', action='store_true', default=False)
    
    # paths to save model
    parser.add_argument('--model_save_path', default='./model_weights')
    parser.add_argument('--no_ckpt', action='store_true', default=False)
    
    # pre-trained models
    parser.add_argument('--pre_trained_paths', nargs='+', default=[])
    parser.add_argument('--sweep', action='store_true', default=False, help='Enable sweep mode')

    return parser.parse_args()


def train_test_supervised_mm_model(args, cfg, dataset_cfg, freeze_encoders=False, limited_k=None):
    experiment_id = generate_experiment_id()
    experiment_info = {
        "dataset": args.dataset,
        "model": 'mm_' + '_'.join([cfg['modalities'][modality]['model'][args.models[i]]['class_name'] for i, modality in enumerate(args.modalities)])
    }
    
    loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, modality='mm_' + '_'.join(args.modalities), dataset=args.dataset, 
        experiment_id=experiment_id, experiment_config_path=args.experiment_config_path, entity='fabiang',
        approach='supervised')
    
    # if using wandb and performing a sweep, overwrite the config params with the sweep params
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment
        # Take model kwargs and merge with experiment config
        for modality, model_name in zip(args.modalities, args.models):
            model_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() 
                              if key.startswith(f'{model_name}.')}
            model_kwargs_dict = flat_to_nested_dict(model_key_values)
            if model_kwargs_dict != {}:
                cfg['modalities'][modality]['model'][model_name]['kwargs'] = {
                        **cfg['modalities'][modality]['model'][model_name]['kwargs'], 
                        **model_kwargs_dict[model_name]
                }
                                
    batch_size = cfg['modalities'][args.modalities[0]]['model'][args.models[0]]['kwargs']['batch_size']
    num_epochs = cfg['experiment']['num_epochs']
    # define transforms for each modality
    train_transforms = {}
    test_transforms = {}
    for i, modality in enumerate(args.modalities):
        model_cfg = cfg['modalities'][modality]['model'][args.models[i]]
        transform_cfg = cfg['modalities'][modality]['transforms']
        model_cfg, transform_cfg = check_sampling_cfg(model_cfg, transform_cfg)
        cur_train_transforms, cur_test_transforms = init_transforms(modality, transform_cfg)
        train_transforms.update(cur_train_transforms)
        test_transforms.update(cur_test_transforms)

    # init datamodule
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=args.modalities, batch_size=batch_size,
        split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
        limited_k=limited_k)

    # init models for each modality and pass to multimodal model class
    models_dict = {}
    for i, modality in enumerate(args.modalities):
        cfg['modalities'][modality]['model'][args.models[i]]['kwargs'] = {**dataset_cfg[modality], **cfg['modalities'][modality]['model'][args.models[i]]['kwargs']}
        if args.ssl_pretrained:
            model = init_ssl_pretrained(cfg['modalities'][modality]['model'][args.models[i]], args.pre_trained_paths[i])
        else:
            model = init_model(cfg['modalities'][modality]['model'][args.models[i]], dataset_cfg['main_metric'], 
                ckpt_path=args.pre_trained_paths[i] if args.pre_trained_paths else None)
        models_dict[modality] = getattr(model, cfg['modalities'][modality]['model'][args.models[i]]['encoder_name'])

    if args.ssl_pretrained or args.pre_trained_paths:
        freeze_encoders = True

    lr_mm = cfg['modalities'][args.modalities[0]]['model'][args.models[0]]['kwargs']['lr']
    model = MultiModalClassifier(models_dict, dataset_cfg[args.modalities[0]]['out_size'], modalities=args.modalities, lr=lr_mm, freeze_encoders=freeze_encoders)

    # setup loggers: tensorboards and/or wandb
    if limited_k is not None:
        approach = 'semi_sup'
    elif args.ssl_pretrained:
        approach = 'ssl_fusion'
    else:
        approach = 'supervised'

    # setup callbacks
    callbacks = setup_callbacks(
        early_stopping_metric = "val_loss",
        early_stopping_mode   = "min",
        class_names           = dataset_cfg["class_names"],
        num_classes           = len(dataset_cfg["class_names"]),
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        metric                = 'val_' + dataset_cfg['main_metric'], 
        dataset               = args.dataset, 
        model                 = 'mm_' + '_'.join(args.models), 
        experiment_id         = experiment_id
    )
    from pytorch_lightning.callbacks import LearningRateMonitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')  
    callbacks.append(lr_monitor)
    
    trainer = Trainer.from_argparse_args(
        args=args,
        logger=loggers_list,
        gpus=1,
        deterministic=True,
        max_epochs=num_epochs,
        default_root_dir='logs', 
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0,
        callbacks=callbacks,
        checkpoint_callback=not args.no_ckpt,
        log_every_n_steps=1
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}
    
    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics

def split_args(args):
    """
    When running wandb in sweep mode, list arguments are passed as singleton lists (e.g. ["inertial skeleton"] instead of ["inertial", "skeleton"]).
    This function fixes them.
    """
    if len(args.modalities) == 1:
        args.modalities = args.modalities[0].split()
    if len(args.models) == 1:
        args.models = args.models[0].split()

    return args

def main():
    args = parse_arguments()
    args = split_args(args)
    cfg = load_yaml_to_dict(args.experiment_config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]
    seed_everything(cfg['experiment']['seed'])
    train_test_supervised_mm_model(args, cfg, dataset_cfg)


if __name__ == '__main__':
    main()
