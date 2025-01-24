import argparse
import random
import importlib
from pytorch_lightning import Trainer, seed_everything

from models.cmc_deep_alignment import ContrastiveMultiviewCodingWithDeepAlignment
from models.multimodal import MultiModalClassifier

from utils.experiment_utils import (dict_to_json, generate_experiment_id,
                                    load_yaml_to_dict)
from utils.training_utils import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    # Dataset and experiment config paths.
    parser.add_argument('--experiment_config_path', required=True)
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    parser.add_argument('--augmentations_path', default='configs/augmentations.yaml', nargs='+')
    
    # Data and models.
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--protocol', default='cross_subject')
    parser.add_argument('--framework', default='cmc_da', choices=["cmc_da"])
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--models', required=True, nargs='+')
    parser.add_argument('--model_save_path', default='./model_weights')

    # used to run only in fine tuning mode
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument('--fine_tuning_ckpt_path', help='Path to a pretrained encoder. Required if running with --fine_tuning.')

    # Other training configs.
    parser.add_argument('--no_ckpt', action='store_true', default=False)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--sweep', action='store_true', default=False, help='Set automatically if running in WandB sweep mode. You do not need to set this manually.')
    
    return parser.parse_args()

def ssl_pre_training(args, modalities, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, experiment_id, loggers_list, loggers_dict):
    seed_everything(experiment_cfg['seed'])
    num_epochs = experiment_cfg['num_epochs_ssl']

    # Handle sweep parameters
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment
        num_epochs = _wandb.config.get("num_epochs_ssl", num_epochs)
        experiment_cfg['num_epochs_ssl'] = num_epochs

        num_epochs_ft = _wandb.config.get('num_epochs_fine_tuning', experiment_cfg['num_epochs_fine_tuning'])
        experiment_cfg['num_epochs_fine_tuning'] = num_epochs_ft

        # SSL hyperparameters
        ssl_cfg['kwargs']['lr_cmc'] = _wandb.config.get("lr_cmc", ssl_cfg['kwargs']['lr_cmc'])
        ssl_cfg['kwargs']['lr_da'] = _wandb.config.get("lr_da", ssl_cfg['kwargs']['lr_da'])
        ssl_cfg['kwargs']['batch_size'] = _wandb.config.get("ssl_batch_size", ssl_cfg['kwargs']['batch_size'])
        ssl_cfg['kwargs']['temperature'] = _wandb.config.get("ssl_temperature", ssl_cfg['kwargs']['temperature'])
        ssl_cfg['kwargs']['hidden_cmc'] = _wandb.config.get("hidden_cmc", ssl_cfg['kwargs']['hidden_cmc'])
        ssl_cfg['kwargs']['hidden_da'] = _wandb.config.get("hidden_da", ssl_cfg['kwargs']['hidden_da'])
        ssl_cfg['kwargs']['beta'] = _wandb.config.get("beta", ssl_cfg['kwargs'].get('beta', 0.5))

        # Fine-tuning hyperparameters
        ft_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('ft_')}
        ft_kwargs_dict = flat_to_nested_dict(ft_key_values)
        if ft_kwargs_dict != {}:
            experiment_cfg['fine_tuning_kwargs'] = ft_kwargs_dict['ft_kwargs']

    # Initialize transforms (+ augmentations) and overwrite sample_length using model definition.
    train_transforms = {}
    test_transforms = {}
    for m in modalities:
        model_cfg = model_cfgs[m]
        transform_cfg = transform_cfgs[m]
        # Update the sample size in transforms to match the sample_length in model config
        transform_cfg = check_sampling_cfg(model_cfg, transform_cfg)
        cur_train_transforms, cur_test_transforms = init_transforms(
            m,
            transform_cfg,
            ssl_random_augmentations=True,
            random_augmentations_dict=augmentation_cfgs[m]
            )
        train_transforms.update(cur_train_transforms)
        test_transforms.update(cur_test_transforms)

    # Initialize datamodule.
    batch_size = ssl_cfg['kwargs']['batch_size']
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=modalities, batch_size=batch_size,
    split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
    ssl=True, n_views=1, num_workers=args.num_workers)

    # Initialize encoders
    global_encoders = {}
    local_encoders = {}
    for m in modalities:
        modality_model_cfg = model_cfgs[m]
        # Global encoder for CMC
        global_cfg = modality_model_cfg['global_encoder']
        global_module = importlib.import_module(f"models.{global_cfg['from_module']}")
        global_class_ = getattr(global_module, global_cfg['class_name'])
        global_kwargs = {**dataset_cfg[m], **global_cfg['kwargs']}
        allowed_keys = global_class_.__init__.__code__.co_varnames
        global_kwargs = {k: v for k, v in global_kwargs.items() if k in allowed_keys}
        global_encoders[m] = global_class_(*global_cfg.get('args', []), **global_kwargs)

        # Local encoder for Deep Alignment
        local_cfg = modality_model_cfg['local_encoder']
        local_module = importlib.import_module(f"models.{local_cfg['from_module']}")
        local_class_ = getattr(local_module, local_cfg['class_name'])
        local_kwargs = {**dataset_cfg[m], **local_cfg['kwargs']}
        allowed_keys = local_class_.__init__.__code__.co_varnames
        local_kwargs = {k: v for k, v in local_kwargs.items() if k in allowed_keys}
        local_encoders[m] = local_class_(*local_cfg.get('args', []), **local_kwargs)

    # Initialize the combined model
    model = ContrastiveMultiviewCodingWithDeepAlignment(
        modalities, 
        global_encoders, 
        local_encoders, 
        **ssl_cfg['kwargs']
    )

    # Setup training callbacks.
    callbacks = setup_callbacks_ssl(
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        dataset               = args.dataset, 
        model                 = "mm_ssl_cmc_da_" + "_".join(args.models), 
        experiment_id         = experiment_id,
    )

    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=[0], deterministic=True, max_epochs=num_epochs, default_root_dir='logs', 
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0, callbacks=callbacks, checkpoint_callback=not args.no_ckpt)
    trainer.fit(model, datamodule)

    return global_encoders, loggers_list, loggers_dict, experiment_id

def fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id, limited_k=None):
    seed_everything(experiment_cfg['seed']) # reset seed for consistency in results
    modalities = args.modalities
    batch_size = experiment_cfg.get('batch_size_fine_tuning', 32)
    num_epochs = experiment_cfg['num_epochs_fine_tuning']
    fine_tuning_kwargs = experiment_cfg.get('fine_tuning_kwargs', {})
    lr = fine_tuning_kwargs.get('lr', 0.001)
    hidden = fine_tuning_kwargs.get('hidden', [256, 128])

    # Initialize the classifier model (MLP trained on concatenated features).
    model = MultiModalClassifier(
        encoders, 
        dataset_cfg['n_classes'], 
        hidden=hidden,
        modalities=modalities, 
        optimizer_name='adam',
        metric_scheduler='accuracy',
        lr=lr,
        freeze_encoders=True
    )

    # Initialize train and test transforms.
    train_transforms = {}
    test_transforms = {}
    for m in modalities:
        cur_train_transforms, cur_test_transforms = init_transforms(m, transform_cfgs[m])
        train_transforms.update(cur_train_transforms)
        test_transforms.update(cur_test_transforms)
    
    # Initialize datamodule.
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=modalities, batch_size=batch_size,
    split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
    num_workers=args.num_workers, limited_k=limited_k)

    callbacks = setup_callbacks(
        early_stopping_metric = "val_loss",
        early_stopping_mode   = "min",
        class_names           = dataset_cfg["class_names"],
        num_classes           = len(dataset_cfg["class_names"]),
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        metric                = 'val_' + dataset_cfg['main_metric'], 
        dataset               = args.dataset, 
        model                 = 'mm_ssl_finetuned_cmc_da_' + "_".join(args.models), 
        experiment_id         = experiment_id
    )

    trainer = Trainer.from_argparse_args(args=args, logger=loggers_list, gpus=[0], deterministic=True, max_epochs=num_epochs, default_root_dir='logs', 
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0, callbacks=callbacks, checkpoint_callback=not args.no_ckpt)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}

    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics

def parse_all_cfgs(args):
    cfg = load_yaml_to_dict(args.experiment_config_path)
    # Initialize dictionaries for configs
    experiment_cfg = cfg.get('experiment', {})
    ssl_cfg = cfg.get('ssl', {})
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]
    model_cfgs = {}
    transform_cfgs = {}
    augmentation_cfgs = {}
    for i, modality in enumerate(args.modalities):
        model_cfgs[modality] = cfg['modalities'][modality]['model'][args.models[i]]
        transform_cfgs[modality] = cfg['modalities'][modality]['transforms']
        augmentation_cfgs[modality] = load_yaml_to_dict(args.augmentations_path[i])

    return experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs

def split_args(args):
    if len(args.modalities) == 1:
        args.modalities = args.modalities[0].split()
    if len(args.models) == 1:
        args.models = args.models[0].split()
    if len(args.augmentations_path) == 1:
        args.augmentations_path = args.augmentations_path[0].split()

    return args

def validate_args(args):
    no_modalities = len(args.modalities)
    if len(args.models) != no_modalities:
        print(f"Supplied {no_modalities} modalities but only {len(args.models)} models!")
        exit(1)
    if len(args.augmentations_path) != no_modalities:
        print(f"Supplied {no_modalities} modalities but only {len(args.augmentations_path)} augmentation config path!")
        exit(1)

    if args.fine_tuning and not args.fine_tuning_ckpt_path:
        print("Need to provide --fine_tuning_ckpt_path if running with --fine_tuning!")
        exit(1)

def init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs,
                 augmentation_cfgs, experiment_id, is_sweep=False):
    num_epochs = experiment_cfg['num_epochs_ssl']
    experiment_info = {
        "dataset": args.dataset,
        "ssl_framework": args.framework,
        "num_epochs_ssl": num_epochs,
        "num_epochs_fine_tuning": experiment_cfg['num_epochs_fine_tuning'],
        "batch_size_fine_tuning": experiment_cfg['batch_size_fine_tuning'],
    }
    for m in modalities:
        modality_model_cfg = model_cfgs[m]
        # Access the encoder_class_name directly
        encoder_class_name = modality_model_cfg['global_encoder']['encoder_class_name']
        experiment_info[f"{m}_encoder"] = encoder_class_name
        experiment_info[f"{m}_augmentations"] = augmentation_cfgs[m]

    loggers_list, loggers_dict = setup_loggers(
        tb_dir="tb_logs",
        experiment_info=experiment_info,
        modality='mm_' + '_'.join(modalities),
        dataset=args.dataset,
        experiment_id=experiment_id,
        experiment_config_path=args.experiment_config_path,
        approach='mm_ssl',
        is_sweep=is_sweep
    )
    return loggers_list, loggers_dict

def run_one_experiment(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs):
    experiment_id = generate_experiment_id()
    modalities = args.modalities
    loggers_list, loggers_dict = init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id, is_sweep=args.sweep)
    
    encoders, loggers_list, loggers_dict, experiment_id = ssl_pre_training(args, modalities, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, experiment_id, loggers_list, loggers_dict)
    result_metrics = fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id)
    return result_metrics

def run_fine_tuning_only(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs):
    experiment_id = generate_experiment_id()
    modalities = args.modalities
    loggers_list, loggers_dict = init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id)

    for m in modalities:
        model_cfgs[m]['kwargs'] = {**dataset_cfg[m], **model_cfgs[m]['kwargs']}
    model = init_ssl_mm_pretrained(modalities, model_cfgs, args.fine_tuning_ckpt_path)
    encoders = model.encoders
    fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id)

def check_sampling_cfg(model_cfg, transform_cfg):
    sample_length = model_cfg['global_encoder'].get('sample_length')
    if sample_length:
        # Update the size parameter in the sampling transform
        for transform in transform_cfg:
            if transform['class_name'] in ['InertialSampler', 'SkeletonSampler']:
                transform['kwargs']['size'] = sample_length
    return transform_cfg 

def main():
    args = parse_arguments()
    args = split_args(args)
    validate_args(args)
    experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs = parse_all_cfgs(args)

    if args.fine_tuning:
        run_fine_tuning_only(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs)
    else:
        run_one_experiment(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs)

if __name__ == '__main__':
    main() 