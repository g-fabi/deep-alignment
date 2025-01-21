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
        optimizer_name='adam',
        lr=0.001
    )

    # Setup training callbacks.
    callbacks = setup_callbacks_ssl(
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        dataset               = args.dataset, 
        model                 = "deep_alignment_only_" + "_".join(args.models), 
        experiment_id         = experiment_id,
    )

    trainer = Trainer.from_argparse_args(args=args,
                                         logger=loggers_list,
                                         gpus=[0],
                                         deterministic=True,
                                         max_epochs=num_epochs,
                                         default_root_dir='logs',
                                         val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0,
                                         callbacks=callbacks)
    trainer.fit(model, datamodule)

    return encoders, loggers_list, loggers_dict, experiment_id

def fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id, limited_k=None):
    seed_everything(experiment_cfg['seed'])  # Reset seed for consistency
    modalities = args.modalities
    batch_size = experiment_cfg['batch_size_fine_tuning']
    num_epochs = experiment_cfg['num_epochs_fine_tuning']
    
    # Initialize the classifier model (MLP trained on concatenated features).
    model = MultiModalClassifier(encoders, dataset_cfg['n_classes'], modalities=modalities, freeze_encoders=True)

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
                                         val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0,
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
        model_cfgs[modality] = cfg['modalities'][modality]['model']
        transform_cfgs[modality] = cfg['modalities'][modality]['transforms']
        augmentation_cfgs[modality] = load_yaml_to_dict(args.augmentations_path[i])

    return experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs

def run_one_experiment(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs):
    experiment_id = generate_experiment_id()
    print(f"Experiment ID: {experiment_id}")
    modalities = args.modalities

    # Initialize loggers (if any)
    loggers_list, loggers_dict = [], {}  # Replace with appropriate logger initialization if needed

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

    # Initialize loggers (if any)
    loggers_list, loggers_dict = [], {}  # Replace with appropriate logger initialization if needed

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