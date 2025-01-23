import argparse
import random
import importlib
from pytorch_lightning import Trainer, seed_everything

from models.deep_alignment_only_model import DeepAlignmentModel
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
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--models', required=True, nargs='+')
    parser.add_argument('--model_save_path', default='./model_weights')

    # used to run only in fine tuning mode
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument('--fine_tuning_ckpt_path', help='Path to a pretrained encoder. Required if running with --fine_tuning.')

    # Other training configs.
    parser.add_argument('--no_ckpt', action='store_true', default=False)
    parser.add_argument('--online-eval', action='store_true', default=False)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--sweep', action='store_true', default=False, help='Set automatically if running in WandB sweep mode. You do not need to set this manually.')
    
    return parser.parse_args()

def ssl_pre_training(args, modalities, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, experiment_id, loggers_list, loggers_dict):
    
    seed_everything(experiment_cfg['seed'])
    num_epochs = experiment_cfg['num_epochs_ssl']
    batch_size = ssl_cfg['kwargs']['batch_size']

    # If using wandb and performing a sweep, overwrite the config params with the sweep params.
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment

        num_epochs_sweep = _wandb.config.get("num_epochs_ssl", num_epochs)
        experiment_cfg['num_epochs_ssl'] = num_epochs_sweep
        num_epochs = num_epochs_sweep
        print(f"Sweep override: num_epochs_ssl = {num_epochs}")

        num_epochs_ft_sweep = _wandb.config.get("num_epochs_fine_tuning", experiment_cfg['num_epochs_fine_tuning'])
        experiment_cfg['num_epochs_fine_tuning'] = num_epochs_ft_sweep
        print(f"Sweep override: num_epochs_fine_tuning = {num_epochs_ft_sweep}")

        # Update SSL hyperparameters
        ssl_cfg['kwargs']['lr'] = _wandb.config.get("ssl_lr", ssl_cfg['kwargs'].get('lr', 0.001))
        ssl_cfg['kwargs']['batch_size'] = _wandb.config.get("ssl_batch_size", ssl_cfg['kwargs'].get('batch_size', 32))
        ssl_cfg['kwargs']['optimizer_name'] = _wandb.config.get("ssl_optimizer_name", ssl_cfg['kwargs'].get('optimizer_name', 'adam'))
        print(f"Sweep override: ssl_cfg['kwargs'] = {ssl_cfg['kwargs']}")

        # Update Fine-tuning hyperparameters
        if 'fine_tuning_kwargs' not in experiment_cfg:
            experiment_cfg['fine_tuning_kwargs'] = {}
        experiment_cfg['fine_tuning_kwargs']['lr'] = _wandb.config.get("ft_lr", experiment_cfg['fine_tuning_kwargs'].get('lr', 0.001))
        experiment_cfg['fine_tuning_kwargs']['hidden'] = _wandb.config.get("ft_hidden", experiment_cfg['fine_tuning_kwargs'].get('hidden', [256, 128]))
        experiment_cfg['fine_tuning_kwargs']['batch_size'] = _wandb.config.get("ft_batch_size", experiment_cfg['fine_tuning_kwargs'].get('batch_size', 32))
        print(f"Sweep override: experiment_cfg['fine_tuning_kwargs'] = {experiment_cfg['fine_tuning_kwargs']}")

        # Update Model Hyperparameters for 'inertial' modality - Local Transformer
        model_cfgs['inertial']['local_transformer']['kwargs']['depth'] = _wandb.config.get("depth_inertial", model_cfgs['inertial']['local_transformer']['kwargs'].get('depth', 5))
        model_cfgs['inertial']['local_transformer']['kwargs']['dim_rep'] = _wandb.config.get("dim_rep_inertial", model_cfgs['inertial']['local_transformer']['kwargs'].get('dim_rep', 256))
        model_cfgs['inertial']['local_transformer']['kwargs']['num_heads'] = _wandb.config.get("num_heads_inertial", model_cfgs['inertial']['local_transformer']['kwargs'].get('num_heads', 8))
        print(f"Sweep override: inertial local_transformer kwargs = {model_cfgs['inertial']['local_transformer']['kwargs']}")

        # Update Model Hyperparameters for 'skeleton' modality - Local Transformer
        model_cfgs['skeleton']['local_transformer']['kwargs']['depth'] = _wandb.config.get("depth_skeleton", model_cfgs['skeleton']['local_transformer']['kwargs'].get('depth', 5))
        model_cfgs['skeleton']['local_transformer']['kwargs']['dim_rep'] = _wandb.config.get("dim_rep_skeleton", model_cfgs['skeleton']['local_transformer']['kwargs'].get('dim_rep', 256))
        model_cfgs['skeleton']['local_transformer']['kwargs']['num_heads'] = _wandb.config.get("num_heads_skeleton", model_cfgs['skeleton']['local_transformer']['kwargs'].get('num_heads', 8))
        print(f"Sweep override: skeleton local_transformer kwargs = {model_cfgs['skeleton']['local_transformer']['kwargs']}")
        for m in modalities:
            print(f"Post-sweep: {m} global_encoder kwargs = {model_cfgs[m]['global_encoder']['kwargs']}")

        # Update local transformer hyperparameters from sweep
        for m in modalities:
            depth_key = f"depth_{m}"
            dim_rep_key = f"dim_rep_{m}"
            num_heads_key = f"num_heads_{m}"
            
            if depth_key in _wandb.config:
                model_cfgs[m]['local_transformer']['kwargs']['depth'] = _wandb.config[depth_key]
            
            if dim_rep_key in _wandb.config:
                model_cfgs[m]['local_transformer']['kwargs']['dim_rep'] = _wandb.config[dim_rep_key]
            
            if num_heads_key in _wandb.config:
                model_cfgs[m]['local_transformer']['kwargs']['num_heads'] = _wandb.config[num_heads_key]

    num_epochs = experiment_cfg['num_epochs_ssl']
    batch_size = ssl_cfg['kwargs'].get('batch_size', batch_size)
    
    # Initialize transforms (+ augmentations) and overwrite sample_length using model definition.
    train_transforms = {}
    test_transforms = {}
    for m in modalities:
        model_cfg = model_cfgs[m]['global_encoder']
        _, transform_cfg = check_sampling_cfg(model_cfg, transform_cfgs[m])
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
    datamodule = init_datamodule(
        data_path=args.data_path,
        dataset_name=args.dataset,
        modalities=modalities,
        batch_size=batch_size,
        split=dataset_cfg['protocols'][args.protocol],
        train_transforms=train_transforms,
        test_transforms=test_transforms,
        ssl=True,
        n_views=1,
        num_workers=args.num_workers
    )

    # Initialize encoders for global and local features
    encoders = {}
    local_transformers = {}
    for m in modalities:
        encoders[m] = init_ssl_encoder(model_cfgs[m]['global_encoder'])
        local_transformers[m] = init_local_transformer(model_cfgs[m]['local_transformer'])

    # Initialize the model.
    model = DeepAlignmentModel(
        modalities=modalities,
        encoders=encoders,
        local_transformers=local_transformers,
        optimizer_name=ssl_cfg['kwargs'].get('optimizer_name', 'adam'),
        lr=ssl_cfg['kwargs'].get('lr', 0.001)
    )

    # Setup training callbacks.
    callbacks = setup_callbacks_ssl(
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        dataset               = args.dataset, 
        model                 = "deep_alignment_only_" + "_".join(args.models), 
        experiment_id         = experiment_id
    )

    trainer = Trainer.from_argparse_args(
        args=args,
        logger=loggers_list,
        gpus=[0],
        deterministic=True,
        max_epochs=num_epochs,
        default_root_dir='logs',
        val_check_interval=1.0,
        log_every_n_steps=1,
        callbacks=callbacks
    )
    trainer.fit(model, datamodule)

    return encoders, loggers_list, loggers_dict, experiment_id

def fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id, limited_k=None):
    
    seed_everything(experiment_cfg['seed'])  # Reset seed for consistency
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
        modalities=modalities,
        freeze_encoders=True,
        lr=lr,
        hidden=hidden
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
        model                 = 'deep_alignment_finetuned_' + "_".join(args.models), 
        experiment_id         = experiment_id
    )

    trainer = Trainer.from_argparse_args(args=args,
                                         logger=loggers_list,
                                         gpus=[0],
                                         deterministic=True,
                                         max_epochs=num_epochs,
                                         default_root_dir='logs', 
                                         val_check_interval = 1.0,
                                         log_every_n_steps=1,
                                         callbacks=callbacks)

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}
    
    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics

def split_args(args):
    """
    When running wandb in sweep mode, list arguments are passed as singleton lists 
    (e.g. ["inertial skeleton"] instead of ["inertial", "skeleton"]).
    This function fixes them.
    """
    if len(args.modalities) == 1:
        args.modalities = args.modalities[0].split()
    if len(args.models) == 1:
        args.models = args.models[0].split()
    if len(args.augmentations_path) == 1:
        args.augmentations_path = args.augmentations_path[0].split()

    return args

def validate_args(args):
    num_modalities = len(args.modalities)
    if len(args.models) != num_modalities:
        print(f"Supplied {num_modalities} modalities but only {len(args.models)} models!")
        exit(1)
    if len(args.augmentations_path) != num_modalities:
        print(f"Supplied {num_modalities} modalities but only {len(args.augmentations_path)} augmentation config paths!")
        exit(1)

    if args.fine_tuning and not args.fine_tuning_ckpt_path:
        print("Need to provide --fine_tuning_ckpt_path if running with --fine_tuning!")
        exit(1)

def parse_all_cfgs(args):
    cfg = load_yaml_to_dict(args.experiment_config_path)
    experiment_cfg = cfg['experiment']
    ssl_cfg = cfg['ssl']
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]

    model_cfgs = {}
    transform_cfgs = {}
    augmentation_cfgs = {}
    for i, modality in enumerate(args.modalities):
        model_cfgs[modality] = {}
        model_cfgs[modality]['global_encoder'] = cfg['modalities'][modality]['model']['global_encoder']
        model_cfgs[modality]['local_transformer'] = cfg['modalities'][modality]['model']['local_transformer']

        transform_cfgs[modality] = cfg['modalities'][modality]['transforms']
        augmentation_cfgs[modality] = load_yaml_to_dict(args.augmentations_path[i])

    return experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs

def init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id, is_sweep=False):
    num_epochs = experiment_cfg['num_epochs_ssl']
    experiment_info = {
        "dataset": args.dataset,
        "ssl_framework": "deep_alignment_only",
        "num_epochs_ssl": num_epochs,
        "num_epochs_fine_tuning": experiment_cfg['num_epochs_fine_tuning'],
        "batch_size_fine_tuning": experiment_cfg['batch_size_fine_tuning'],
        "wandb_project": experiment_cfg.get('wandb', {}).get('project', 'default_project'),
        "wandb_entity": experiment_cfg.get('wandb', {}).get('entity', None)
    }
    for m in modalities:
        experiment_info[f"{m}_encoder"] = model_cfgs[m]['global_encoder']['class_name']
        experiment_info[f"{m}_augmentations"] = augmentation_cfgs[m]
    
    # Flatten the configurations to get concise parameter names
    flat_experiment_info = nested_to_flat_dict(experiment_info)
    
    loggers_list, loggers_dict = setup_loggers(
        logger_names=['tensorboard', 'wandb'],
        tb_dir="tb_logs",
        experiment_info=flat_experiment_info,
        modality='mm_' + '_'.join(modalities),
        dataset=args.dataset,
        experiment_id=experiment_id,
        experiment_config_path=args.experiment_config_path,
        approach='mm_ssl',
        is_sweep=is_sweep
    )

    # Log additional information where model_cfgs is available
    wandb_logger = loggers_dict.get('wandb')
    if wandb_logger:
        wandb_logger.experiment.config.update({
            "experiment_id": experiment_id,
            "model_name": model_cfgs['inertial']['local_transformer']['class_name']
        }, allow_val_change=True)
    
    return loggers_list, loggers_dict

def run_one_experiment(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs):
    experiment_id = generate_experiment_id()
    print(f"Experiment ID: {experiment_id}")
    modalities = args.modalities

    # Initialize loggers
    loggers_list, loggers_dict = init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id, is_sweep=args.sweep)

    encoders, loggers_list, loggers_dict, experiment_id = ssl_pre_training(
        args, modalities, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, 
        transform_cfgs, augmentation_cfgs, experiment_id, loggers_list, loggers_dict
    )
    result_metrics = fine_tuning(
        args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, 
        loggers_list, loggers_dict, experiment_id
    )
    return result_metrics

def run_fine_tuning_only(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs):
    experiment_id = generate_experiment_id()
    modalities = args.modalities

    # Initialize loggers
    loggers_list, loggers_dict = init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id)

    # Initialize the model from a checkpoint
    model = init_ssl_mm_pretrained(args.modalities, model_cfgs, args.fine_tuning_ckpt_path)
    encoders = model.encoders

    fine_tuning(
        args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, 
        loggers_list, loggers_dict, experiment_id
    )

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