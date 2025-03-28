import os
# Disable only process monitoring env vars
os.environ['PL_DISABLE_FORK'] = '1'
os.environ['WANDB_WATCH'] = 'false'
os.environ['WANDB_DISABLE_STATS'] = '1'
import argparse
import random
import pytorch_lightning
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor

# Disable GPU stats monitoring by replacing the monitor with a no-op version
class NoOpGPUStatsMonitor(GPUStatsMonitor):
    def on_train_start(self, *args, **kwargs): pass
    def on_train_epoch_start(self, *args, **kwargs): pass
    def on_train_batch_start(self, *args, **kwargs): pass
    def on_validation_epoch_start(self, *args, **kwargs): pass
    def on_test_epoch_start(self, *args, **kwargs): pass
    def on_train_epoch_end(self, *args, **kwargs): pass
    def on_validation_epoch_end(self, *args, **kwargs): pass
    def on_test_epoch_end(self, *args, **kwargs): pass

pytorch_lightning.callbacks.gpu_stats_monitor.GPUStatsMonitor = NoOpGPUStatsMonitor
# Disable memory functions that might call system tools
pytorch_lightning.utilities.memory.get_memory_stats = lambda: {}

from pytorch_lightning import Trainer, seed_everything
from models.cmc import ContrastiveMultiviewCoding
from models.cmc_cvkm import ContrastiveMultiviewCodingCVKM
from models.multimodal import MultiModalClassifier
from models.similarity_metrics.latent_space_similarity import LatentSpaceSimilarity
from models.deep_alignment import DeepAlignmentLoss, DeepAlignmentModel
from models.cmc_da import ContrastiveMultiviewCodingDA
import torch
import json
import multiprocessing
import torch.multiprocessing

from utils.experiment_utils import (dict_to_json, generate_experiment_id,
                                    load_yaml_to_dict)
from utils.training_utils import *

import psutil
original_process_iter = psutil.process_iter
def no_op_process_iter(*args, **kwargs):
    return []
psutil.process_iter = no_op_process_iter
multiprocessing.set_start_method('spawn', force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

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
    parser.add_argument('--framework', default='cmc', choices=["cmc", "cmc-cmkm", "da", "cmc-da"])
    parser.add_argument('--modalities', required=True, nargs='+')
    parser.add_argument('--models', required=True, nargs='+')
    parser.add_argument('--model_save_path', default='./model_weights')
    
    # Pretrained encoders (required for cmc-cmkm framework, if using latent space similarity)
    parser.add_argument('--cmkm_pretrained_encoders_config_paths', nargs='+')
    parser.add_argument('--cmkm_pretrained_encoder_paths', nargs='+')

    # used to run only in fine tuning mode
    parser.add_argument('--fine_tuning', action='store_true')
    parser.add_argument('--fine_tuning_ckpt_path', help='Path to a pretrained encoder. Required if running with --fine_tuning.')

    # Other training configs.
    parser.add_argument('--no_ckpt', action='store_true', default=False)
    parser.add_argument('--online-eval', action='store_true', default=False)
    parser.add_argument('--num-workers', default=1, type=int)
    parser.add_argument('--sweep', action='store_true', default=False, help='Set automatically if running in WandB sweep mode. You do not need to set this manually.')
    
    return parser.parse_args()

def ssl_pre_training(args, modalities, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, experiment_id, loggers_list, loggers_dict):
    seed_everything(experiment_cfg['seed'])
    num_epochs = experiment_cfg['num_epochs_ssl']

    # if using wandb and performing a sweep, overwrite the config params with the sweep params.
    if args.sweep:
        _wandb = loggers_dict['wandb'].experiment

        # Take some specific parameters.
        if "num_epochs_ssl" in _wandb.config:
            num_epochs = _wandb.config["num_epochs_ssl"]

        # Take SSL model kwargs and merge with experiment config.
        ssl_key_values = {key: _wandb.config[key] for key in _wandb.config.keys() if key.startswith('ssl.')}
        ssl_kwargs_dict = flat_to_nested_dict(ssl_key_values)
        if ssl_kwargs_dict != {}:
            ssl_cfg['kwargs'] = {**ssl_cfg['kwargs'], **ssl_kwargs_dict['ssl']}
            
        # After applying sweep parameters, save the full configuration
        full_config = {
            "experiment": experiment_cfg,
            "ssl": ssl_cfg,
            "modalities": {m: model_cfgs[m] for m in modalities}
        }
        with open(os.path.join(_wandb.dir, "full_config.json"), 'w') as f:
            json.dump(full_config, f, indent=2, default=str)

    # Initialize transforms (+ augmentations)
    if args.framework == 'da':
        train_transforms = {}
        test_transforms = {}
        for m in modalities:
            _, transform_cfg = check_sampling_cfg(model_cfgs[m], transform_cfgs[m])
            cur_train_transforms, cur_test_transforms = init_transforms(m, transform_cfg,
                                                                        ssl_random_augmentations=False,
                                                                        random_augmentations_dict={})
            train_transforms.update(cur_train_transforms)
            test_transforms.update(cur_test_transforms)
    else:
        train_transforms = {}
        test_transforms = {}
        for m in modalities:
            _, transform_cfg = check_sampling_cfg(model_cfgs[m], transform_cfgs[m])
            cur_train_transforms, cur_test_transforms = init_transforms(m, transform_cfg,
                                                                        ssl_random_augmentations=True,
                                                                        random_augmentations_dict=augmentation_cfgs[m])
            train_transforms.update(cur_train_transforms)
            test_transforms.update(cur_test_transforms)

    # Initialize datamodule.
    batch_size = ssl_cfg['kwargs']['batch_size']
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=modalities, batch_size=batch_size,
    split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
    ssl=True, n_views=1, num_workers=args.num_workers)

    # Merge general model params with dataset-specific model params.
    for m in modalities:
        model_cfgs[m]['kwargs'] = {**dataset_cfg[m], **model_cfgs[m]['kwargs']}

    # Initialize encoders and SSL framework.
    encoders = {}
    for m in modalities:
        encoders[m] = init_ssl_encoder(model_cfgs[m])

    if args.framework == 'cmc':
        model = ContrastiveMultiviewCoding(modalities, encoders, **ssl_cfg['kwargs'])
    elif args.framework == 'cmc-cmkm':
        similarity_metrics = init_similarity_metrics(args, modalities, ssl_cfg, dataset_cfg)
        model = ContrastiveMultiviewCodingCVKM(modalities, encoders, similarity_metrics, **ssl_cfg['kwargs'])
    elif args.framework == 'da':
        if len(modalities) != 2:
            raise ValueError("Deep Alignment pretraining requires exactly 2 modalities.")
        loss_fn = DeepAlignmentLoss(
            beta=ssl_cfg.get('da_beta', 0.5),
            iteration=ssl_cfg.get('da_iteration', 50),
            weight_spatial=ssl_cfg.get('weight_spatial', 1.0),
            weight_temporal=ssl_cfg.get('weight_temporal', 1.0)
        )
        model = DeepAlignmentModel(modalities, encoders, loss_fn, **ssl_cfg['kwargs'])
    elif args.framework == 'cmc-da':
        if len(modalities) != 2:
            raise ValueError("CMC_DA pretraining requires exactly 2 modalities.")
        if "optimizer_name_ssl" in ssl_cfg["kwargs"]:
            ssl_cfg["kwargs"]["optimizer_name"] = ssl_cfg["kwargs"].pop("optimizer_name_ssl")
        model = ContrastiveMultiviewCodingDA(modalities, encoders, **ssl_cfg['kwargs'])

    # Setup training callbacks.
    callbacks = setup_callbacks_ssl(
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        dataset               = args.dataset, 
        model                 = "mm_ssl_" + args.framework + '_' + "_".join(args.models), 
        experiment_id         = experiment_id,
    )
    
    early_stopping_callback = next((cb for cb in callbacks if isinstance(cb, EarlyStopping)), None)

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
        log_every_n_steps=float('inf'),
        progress_bar_refresh_rate=1,
        process_position=0,
        track_grad_norm=-1,
        num_sanity_val_steps=0
    )
    
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"SSL training encountered an error: {str(e)}")
        was_early_stopped = True
        if args.sweep and 'wandb' in loggers_dict:
            loggers_dict['wandb'].experiment.log({
                "early_stopped": True,
                "training_error": str(e)
            })
        return model.encoders, loggers_list, loggers_dict, experiment_id, was_early_stopped
    
    was_early_stopped = False
    if early_stopping_callback and early_stopping_callback.stopped_epoch > 0:
        was_early_stopped = True
        print(f"SSL pre-training was early stopped at epoch {early_stopping_callback.stopped_epoch}")
        
        if args.sweep and 'wandb' in loggers_dict:
            loggers_dict['wandb'].experiment.log({"early_stopped": True, 
                                                 "stopped_epoch": early_stopping_callback.stopped_epoch})

    return model.encoders, loggers_list, loggers_dict, experiment_id, was_early_stopped

def fine_tuning(args, experiment_cfg, dataset_cfg, transform_cfgs, encoders, loggers_list, loggers_dict, experiment_id, limited_k=None):
    seed_everything(experiment_cfg['seed']) # reset seed for consistency in results
    modalities = args.modalities
    batch_size = experiment_cfg['batch_size_fine_tuning']
    num_epochs = experiment_cfg['num_epochs_fine_tuning']
    
    # Initialize the classifier model (MLP trained on concatenated features).
    # To bring the features for the different modalities to the same size,
    # each modality's features will be passed through an additional MLP. 
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
        model                 = 'mm_ssl_finetuned_' + args.framework + '_' + "_".join(args.models), 
        experiment_id         = experiment_id
    )

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
        log_every_n_steps=10,
        progress_bar_refresh_rate=10,
        process_position=0,
        track_grad_norm=0,
        num_sanity_val_steps=0
    )

    trainer.fit(model, datamodule)
    
    # Get the checkpoint callback to find the best model path
    checkpoint_callback = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint)), None)
    early_stopping_callback = next((cb for cb in callbacks if isinstance(cb, EarlyStopping)), None)
    
    # Determine which checkpoint to use for testing
    if checkpoint_callback and checkpoint_callback.best_model_path:
        test_ckpt_path = checkpoint_callback.best_model_path
    else:
        test_ckpt_path = None
        
    if early_stopping_callback and early_stopping_callback.stopped_epoch > 0:
        if 'wandb' in loggers_dict:
            loggers_dict['wandb'].experiment.log({
                "fine_tuning_early_stopped": True,
                "fine_tuning_stopped_epoch": early_stopping_callback.stopped_epoch
            })
    
    trainer.test(model, datamodule, ckpt_path=test_ckpt_path)

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}
    
    # Log test metrics to wandb
    if 'wandb' in loggers_dict:
        test_metrics = {f"test_{k}": v for k, v in metrics.items()}
        loggers_dict['wandb'].experiment.log(test_metrics)
        loggers_dict['wandb'].experiment.finish()

    return metrics

def init_similarity_metrics(args, modalities, ssl_cfg, dataset_cfg):
    SUPPORTED_SIMILARITY_METRICS = ["pretrained_encoder", "raw_similarities"]
    SUPPORTED_MODALITIES = ["inertial", "skeleton"]

    cmkm_config = ssl_cfg['kwargs']['cmkm_config']
    
    if cmkm_config['similarity_metric'] not in SUPPORTED_SIMILARITY_METRICS:
        print("cmkm_config['similarity_metric'] must be one of 'pretrained_encoder' or 'raw_similarities'!")
        exit(1)

    if set(modalities) != set(SUPPORTED_MODALITIES):
        print(f"CVKM supported modalities are {SUPPORTED_MODALITIES}!")
        exit(1)

    similarity_metrics = {}

    if cmkm_config['similarity_metric'] == "pretrained_encoder":
        for i, m in enumerate(modalities):
            pretrained_encoder_cfg = load_yaml_to_dict(args.cmkm_pretrained_encoders_config_paths[i])['modalities'][m]['model'][args.models[i]]
            pretrained_encoder_cfg['kwargs'] = {**dataset_cfg[m], **pretrained_encoder_cfg['kwargs']}
            pretrained_model = init_ssl_pretrained(pretrained_encoder_cfg, args.cmkm_pretrained_encoder_paths[i])
            pretrained_encoder = pretrained_model.encoder
            pretrained_encoder.freeze()
            similarity_metrics[m] = LatentSpaceSimilarity(m, pretrained_encoder)

    return similarity_metrics

def parse_all_cfgs(args):
    cfg = load_yaml_to_dict(args.experiment_config_path)
    experiment_cfg = cfg['experiment']
    ssl_cfg = cfg['ssl']
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
    """
    When running wandb in sweep mode, list arguments are passed as singleton lists (e.g. ["inertial skeleton"] instead of ["inertial", "skeleton"]).
    This function fixes them.
    """
    if len(args.modalities) == 1:
        args.modalities = args.modalities[0].split()
    if args.cmkm_pretrained_encoders_config_paths and len(args.cmkm_pretrained_encoders_config_paths) == 1:
        args.cmkm_pretrained_encoders_config_paths = args.cmkm_pretrained_encoders_config_paths[0].split()
    if args.cmkm_pretrained_encoder_paths and len(args.cmkm_pretrained_encoder_paths) == 1:
        args.cmkm_pretrained_encoder_paths = args.cmkm_pretrained_encoder_paths[0].split()
    if len(args.models) == 1:
        args.models = args.models[0].split()
    if len(args.augmentations_path) == 1:
        args.augmentations_path = args.augmentations_path[0].split()

    return args

def init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id):
    num_epochs = experiment_cfg['num_epochs_ssl']
    if args.framework == "cmc-cmkm":
        flat_cmkm_config = nested_to_flat_dict({"ssl": {"cmkm_config": ssl_cfg['kwargs']['cmkm_config']}})
    else:
        flat_cmkm_config = {}
    experiment_info = {
        "dataset": args.dataset,
        "ssl_framework": args.framework,
        "num_epochs_ssl": num_epochs,
        "num_epochs_fine_tuning": experiment_cfg['num_epochs_fine_tuning'],
        "batch_size_fine_tuning": experiment_cfg['batch_size_fine_tuning'],
        **flat_cmkm_config
    }
    for m in modalities:
        experiment_info[f"{m}_encoder"] = model_cfgs[m]['encoder_class_name']
        experiment_info[f"{m}_augmentations"] = augmentation_cfgs[m]
    
    loggers_list, loggers_dict = setup_loggers(
        tb_dir="tb_logs", 
        experiment_info=experiment_info, 
        modality='mm_' + '_'.join(modalities), 
        dataset=args.dataset, 
        experiment_id=experiment_id, 
        experiment_config_path=args.experiment_config_path,
        entity='fabiang',
        approach='mm_ssl'
    )
    return loggers_list, loggers_dict

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

def run_one_experiment(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, existing_loggers=None):
    experiment_id = generate_experiment_id()
    modalities = args.modalities
    
    # Use existing loggers if provided
    if existing_loggers:
        loggers_list, loggers_dict = existing_loggers
    else:
        loggers_list, loggers_dict = init_loggers(args, modalities, experiment_cfg, ssl_cfg, model_cfgs, augmentation_cfgs, experiment_id)
    
    encoders, loggers_list, loggers_dict, experiment_id, was_early_stopped = ssl_pre_training(args, modalities, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, experiment_id, loggers_list, loggers_dict)
    
    # Log if early stopping occurred, but still proceed with fine-tuning
    if was_early_stopped:
        if args.sweep and 'wandb' in loggers_dict:
            loggers_dict['wandb'].experiment.log({"early_stopped_but_finetuned": True})
    
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

def main():
    args = parse_arguments()
    args = split_args(args)
    validate_args(args)
    experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs = parse_all_cfgs(args)
    
    existing_loggers = None
    if args.sweep:
        experiment_id = generate_experiment_id()
        experiment_info = {
            "dataset": args.dataset,
            "ssl_framework": args.framework,
            "model": 'mm_ssl_' + '_'.join(['placeholder' for _ in args.modalities])
        }
        loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, 
                                        modality='mm_' + '_'.join(args.modalities), 
                                        dataset=args.dataset, experiment_id=experiment_id, 
                                        experiment_config_path=args.experiment_config_path, 
                                        entity='fabiang', approach='mm_ssl')
        existing_loggers = (loggers_list, loggers_dict)
        
        if 'seed' in loggers_dict['wandb'].experiment.config:
            experiment_cfg['seed'] = loggers_dict['wandb'].experiment.config['seed']
            print(f"Using seed from wandb config: {experiment_cfg['seed']}")
    
    seed_everything(experiment_cfg['seed'])
    
    if args.fine_tuning:
        run_fine_tuning_only(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs)
    else:
        run_one_experiment(args, experiment_cfg, ssl_cfg, dataset_cfg, model_cfgs, transform_cfgs, augmentation_cfgs, existing_loggers)

if __name__ == '__main__':
    main()