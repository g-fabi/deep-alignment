#training_utils.py

import importlib
import itertools
import os
import shutil

import torch
from models.cmc import ContrastiveMultiviewCoding
from models.simclr_um import SimCLRUnimodal
from torchvision import transforms
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from callbacks.log_classifier_metrics import LogClassifierMetrics
from callbacks.log_confusion_matrix import LogConfusionMatrix
from data_modules.constants import DATASET_PROPERTIES
from transforms.augmentation_utils import compose_random_augmentations
from utils.experiment_utils import load_yaml_to_dict

def init_transforms(modality, transforms_cfg, ssl_random_augmentations=False, random_augmentations_dict={}):
    train = []
    test = []
    for t in transforms_cfg:
        module = importlib.import_module(f"transforms.{t['from_module']}")
        class_ = getattr(module, t['class_name'])

        if "kwargs" in t:
            transform = class_(**t['kwargs'])
        else:
            transform = class_()

        train.append(transform)
        if t['in_test']:
            test.append(transform)
    if ssl_random_augmentations:
        train.extend(compose_random_augmentations(modality, random_augmentations_dict))
    composed_train_transform = transforms.Compose(train)
    composed_test_transform = transforms.Compose(test)

    train_transforms = { modality: composed_train_transform }
    test_transforms = { modality: composed_test_transform }

    return train_transforms, test_transforms

  
def init_datamodule(data_path, dataset_name, modalities, batch_size,
                    split, train_transforms, test_transforms,
                    ssl = False, n_views = 2, num_workers = 1, limited_k=None):
    dataset_properties = DATASET_PROPERTIES[dataset_name]
    print(dataset_properties)
    data_module = dataset_properties.datamodule_class(path = data_path, modalities = modalities, batch_size = batch_size, 
        split=split, train_transforms = train_transforms, test_transforms = test_transforms,
        ssl = ssl, n_views = n_views, num_workers = num_workers, limited_k = limited_k)
    return data_module

  
def init_model(model_cfg, metric_scheduler='accuracy', ckpt_path=None):
    module = importlib.import_module(f"models.{model_cfg['from_module']}")
    class_ = getattr(module, model_cfg['class_name'])
    if ckpt_path is None:
        return class_(*model_cfg['args'], **model_cfg['kwargs'], metric_scheduler=metric_scheduler)
    else:
        return class_.load_from_checkpoint(ckpt_path, strict=False)


def init_ssl_pretrained(model_cfg, ckpt_path):
    encoder = init_ssl_encoder(model_cfg)
    class_ = SimCLRUnimodal('inertial', encoder, encoder.out_size)
    return class_.load_from_checkpoint(ckpt_path, encoder=encoder, mlp_in_size=encoder.out_size, strict=False)

def init_ssl_mm_pretrained(modalities, model_cfgs, ckpt_path):
    encoders = {}
    for m in modalities:
        encoders[m] = init_ssl_encoder(model_cfgs[m])

    ckpt = torch.load(ckpt_path, map_location='cpu')
    h = ckpt["hyper_parameters"]
    model = ContrastiveMultiviewCoding(
        modalities,
        encoders,
        hidden=h["hidden"],
        lr=h["lr"],
        batch_size=h["batch_size"],
        temperature=h["temperature"],
        optimizer_name_ssl=h["optimizer_name_ssl"],
    )
    model = model.load_from_checkpoint(
        ckpt_path,
        modalities=modalities,
        encoders=encoders,
        strict=False
    )
    return model

def init_ssl_encoder(encoder_cfg, ckpt_path=None):
    """
    Initialize SSL encoder based on the configuration.

    Args:
        encoder_cfg (dict): Configuration dictionary for the encoder.
        ckpt_path (str, optional): Path to a checkpoint file. Defaults to None.

    Returns:
        nn.Module: Initialized encoder.
    """
    module = importlib.import_module(f"models.{encoder_cfg['from_module']}")
    class_ = getattr(module, encoder_cfg['class_name'])
    if ckpt_path is None:
        return class_(*encoder_cfg.get('args', []), **encoder_cfg.get('kwargs', {}))
    else:
        return class_.load_from_checkpoint(ckpt_path)



def parse_splits(dataset_configs):
    return dataset_configs['splits']['train'], dataset_configs['splits']['val'], dataset_configs['splits']['test']


def setup_tb_logger(dir, name):
    return loggers.TensorBoardLogger(dir, name=name)


def setup_wandb_logger(experiment_info, modality, dataset, experiment_id, approach='supervised'):
    entity = experiment_info.get('wandb_entity', None)
    project = experiment_info.get('wandb_project', f"{approach}-{modality}-{dataset}")    
    return loggers.WandbLogger(
        config=experiment_info,
        entity=entity,
        project=project,
        name=experiment_id,
        id=experiment_id)


def setup_loggers(logger_names=['tensorboard', 'wandb'], tb_dir=None, experiment_info=None, modality=None, dataset=None, 
                  experiment_id=None, approach='supervised', experiment_config_path=None, is_sweep=False):
    loggers_list = []
    loggers_dict = {}
    if 'tensorboard' in logger_names:
        tb_logger = setup_tb_logger(tb_dir, experiment_id)
        loggers_list.append(tb_logger)
        loggers_dict['tensorboard'] = tb_logger
    if 'wandb' in logger_names:
        if is_sweep:
            wandb_logger = loggers.WandbLogger(
                config=experiment_info,
                entity=experiment_info.get('wandb_entity', None),
                project=experiment_info.get('wandb_project', f"{approach}-{modality}-{dataset}"),
                name=experiment_id,
                id=experiment_id
            )
        else:
            wandb_logger = setup_wandb_logger(experiment_info, modality, dataset, experiment_id, approach)
        loggers_list.append(wandb_logger)
        loggers_dict['wandb'] = wandb_logger
        if experiment_config_path:
            shutil.copy(experiment_config_path, os.path.join(wandb_logger.experiment.dir, "experiment_config.yaml"))
    return loggers_list, loggers_dict


def setup_early_stopping_callback(metric, min_delta=0.00, patience=50, mode="min"):
    return EarlyStopping(monitor=metric, min_delta=min_delta, patience=patience, verbose=False, mode=mode)


def setup_confusion_matrix_logger(class_names):
    return LogConfusionMatrix(class_names)


def setup_classifier_metrics_logger(num_classes, metric_names=['accuracy', 'f1-score', 'precision', 'recall'], average='macro'):
    return LogClassifierMetrics(num_classes, metric_names, average=average)


def setup_model_checkpoint_callback(model_weights_path, metric, dataset, model, experiment_id):
    return ModelCheckpoint(
        monitor=metric, 
        dirpath=os.path.join(model_weights_path, f"{dataset}-{model}-{experiment_id}"),
        filename="{epoch}",
        save_top_k=1,
        mode="max"
    )	


def setup_model_checkpoint_callback_last(model_weights_path, dataset, model, experiment_id):
    return ModelCheckpoint(
        save_last=True,
        dirpath=os.path.join(model_weights_path, f"{dataset}-{model}-{experiment_id}"),
        filename="{epoch}"
    )	


def setup_callbacks(early_stopping_metric, early_stopping_mode, class_names, num_classes, no_ckpt, model_weights_path, metric, dataset, model, experiment_id):
    callbacks = []
    callbacks.append(setup_early_stopping_callback(early_stopping_metric, mode=early_stopping_mode))
    callbacks.append(setup_confusion_matrix_logger(class_names))
    callbacks.append(setup_classifier_metrics_logger(num_classes))
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback(model_weights_path, metric, dataset, model, experiment_id))
    return callbacks


def setup_callbacks_ssl(no_ckpt, model_weights_path, dataset, model, experiment_id):
    callbacks = []
    if not no_ckpt:
        callbacks.append(setup_model_checkpoint_callback_last(model_weights_path, dataset, model, experiment_id))
    return callbacks


def get_tuning_grid_list(tuning_config_path, modality, model):
    tuning_config = load_yaml_to_dict(tuning_config_path)[modality][model]['kwargs']
    return [dict(zip(tuning_config.keys(), x)) for x in itertools.product(*tuning_config.values())]


def check_sampling_cfg(model_cfg, transform_cfg):
    sample_length = model_cfg.get('max_seq_len') or model_cfg.get('maxlen')
    if sample_length is not None:
        for i, transform in enumerate(transform_cfg):
            if ('transform_name' in transform
                and transform['transform_name'] == 'sampling'
                and transform['kwargs']['size'] != sample_length):
                transform['kwargs']['size'] = sample_length
    return model_cfg, transform_cfg

def flat_key_to_dict(flat_key, value):
    """
    Example input: "augmentations_configs.scaling.apply", True
    Example output: {'augmentations_configs': {'scaling': {'apply': True}}}
    """
    tokens = flat_key.split('.')
    if len(tokens) == 1:
        return {flat_key: value}
    
    last_key = tokens[-1]
    last_dict = {last_key: value}
    for inner_key in tokens[-2::-1]:
        last_dict = {inner_key: last_dict}

    return last_dict

def deep_merge_dicts(dict1, dict2):
    """
    Performs a deep merge of dict1 and dict2, with leaf values from dict2 overwriting dict1 if needed.
    """
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(deep_merge_dicts(dict1[k], dict2[k])))
            else:
                # If one of the values is not a dict, value from second dict overrides one in first and we move on.
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])

def flat_to_nested_dict(flat_dict):
    """
    Example input:
        {
            "augmentations.scaling.apply": True,
            "augmentations.scaling.parameters.min_p": 0.5,
            "augmentations.scaling.parameters.max_p": 0.8
        }
    Example output:
        {
            "augmentations": {
                "scaling": {
                    "apply": True,
                    "parameters": {
                        "min_p": 0.5,
                        "max_p": 0.8
                    }
                }
            }
        }
    """
    nested_dict = {}
    for key in flat_dict:
        nested_inner = flat_key_to_dict(key, flat_dict[key])
        nested_dict = dict(deep_merge_dicts(nested_dict, nested_inner))
    return nested_dict

def nested_to_flat_dict(nested_dict):
    out = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            val = [val]
        if isinstance(val, list):
            for subdict in val:
                deeper = nested_to_flat_dict(subdict).items()
                out.update({key + '.' + key2: val2 for key2, val2 in deeper})
        else:
            out[key] = val
    return out


#FABIAN: UTIL FUNCTIONS FOR DEEP ALIGNMENT


def init_local_transformer(model_cfg, ckpt_path=None):
    module = importlib.import_module(f"models.{model_cfg['from_module']}")
    class_name = model_cfg['class_name']
    class_ = getattr(module, class_name)
    kwargs = model_cfg.get('kwargs', {}).copy()
    kwargs.pop('sample_length', None)
    if ckpt_path is None:
        return class_(*model_cfg.get('args', []), **kwargs)
    else:
        return class_.load_from_checkpoint(ckpt_path)

def init_encoders(modalities, model_cfgs, ckpt_paths=None):
    encoders = {}
    for m in modalities:
        model_cfg = model_cfgs[m]['global_encoder']

        # Print the kwargs being passed
        print(f"Initializing encoder for modality '{m}' with kwargs: {model_cfg.get('kwargs', {})}")

        module = importlib.import_module(f"{model_cfg['from_module']}")
        class_ = getattr(module, model_cfg['class_name'])
        encoder = class_(*model_cfg.get('args', []), **model_cfg.get('kwargs', {}))
        if ckpt_paths and ckpt_paths.get(m):
            encoder.load_state_dict(torch.load(ckpt_paths[m]))
        encoders[m] = encoder
    return encoders
