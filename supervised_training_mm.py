import argparse
import importlib

from pytorch_lightning import Trainer, seed_everything

from models.multimodal import MultiModalClassifier
from models.cmc_deep_alignment import ContrastiveMultiviewCodingWithDeepAlignment

from utils.experiment_utils import generate_experiment_id, load_yaml_to_dict
from utils.training_utils import *
from callbacks.log_classifier_metrics import LogClassifierMetrics


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
    parser.add_argument('--num_workers', type=int, default=4)
    # ssl pre-trained 
    parser.add_argument('--ssl_pretrained', action='store_true', default=False)
    
    # paths to save model
    parser.add_argument('--model_save_path', default='./model_weights')
    parser.add_argument('--no_ckpt', action='store_true', default=False)
    
    # pre-trained models
    parser.add_argument('--pre_trained_paths', nargs='+', default=[])

    return parser.parse_args()


def init_model(args, cfg, dataset_cfg, encoders_dict, freeze_encoders=False):
    """Initialize the appropriate model based on the framework specified in config"""
    framework = cfg['experiment'].get('framework', 'supervised')  # Default to supervised
    
    if framework == 'cmc_da':
        local_encoders = {}
        for i, modality in enumerate(args.modalities):
            module = importlib.import_module(f"models.{cfg['modalities'][modality]['model'][args.models[i]]['from_module']}")
            model_class = getattr(module, cfg['modalities'][modality]['model'][args.models[i]]['class_name'])
            local_model = model_class(
                *cfg['modalities'][modality]['model'][args.models[i]]['args'],
                **cfg['modalities'][modality]['model'][args.models[i]]['kwargs']
            )
            local_encoders[modality] = local_model

        ssl_cfg = cfg['experiment'].get('ssl', {})
        return ContrastiveMultiviewCodingWithDeepAlignment(
            modalities=args.modalities,
            global_encoders=encoders_dict,
            local_encoders=local_encoders,
            hidden_cmc=ssl_cfg.get('hidden_cmc', [256, 128]),
            hidden_da=ssl_cfg.get('hidden_da', [256, 128]),
            lr_cmc=cfg['experiment']['lr'],
            lr_da=cfg['experiment']['lr'],
            batch_size=cfg['experiment']['batch_size'],
            temperature=ssl_cfg.get('temperature', 0.1),
            optimizer_name_ssl='adam',
            beta=ssl_cfg.get('beta', 0.5)
        )
    else:
        return MultiModalClassifier(
            encoders_dict, 
            dataset_cfg[args.modalities[0]]['out_size'], 
            modalities=args.modalities, 
            freeze_encoders=freeze_encoders
        )


def train_test_supervised_mm_model(args, cfg, dataset_cfg, freeze_encoders=False, limited_k=None):
    experiment_id = generate_experiment_id()

    batch_size = cfg['experiment']['batch_size']
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
        limited_k=limited_k, num_workers=args.num_workers)

    # init models for each modality and pass to multimodal model class
    models_dict = {}
    for i, modality in enumerate(args.modalities):
        if args.ssl_pretrained:
            model = init_ssl_pretrained(cfg['modalities'][modality]['model'][args.models[i]], args.pre_trained_paths[i])
        else:
            module = importlib.import_module(f"models.{cfg['modalities'][modality]['model'][args.models[i]]['from_module']}")
            model_class = getattr(module, cfg['modalities'][modality]['model'][args.models[i]]['class_name'])
            model = model_class(
                *cfg['modalities'][modality]['model'][args.models[i]]['args'],
                **cfg['modalities'][modality]['model'][args.models[i]]['kwargs']
            )

        encoder_name = cfg['modalities'][modality]['model'][args.models[i]].get('encoder_name', None)
        if encoder_name in [None, '', 'none']:
            encoders_dict[modality] = model
        else:
            encoders_dict[modality] = getattr(model, encoder_name)

    freeze_encoders = False

    model = init_model(args, cfg, dataset_cfg, encoders_dict, freeze_encoders)

    experiment_info = {
        "dataset": args.dataset,
        "model": 'mm_' + '_'.join([cfg['modalities'][modality]['model'][args.models[i]]['class_name'] for i, modality in enumerate(args.modalities)])
    }

    # setup loggers: tensorboards and/or wandb
    if limited_k is not None:
        approach = 'semi_sup'
    elif args.ssl_pretrained:
        approach = 'ssl_fusion'
    else:
        approach = 'supervised'

    loggers_list, loggers_dict = setup_loggers(tb_dir="tb_logs", experiment_info=experiment_info, modality='mm_' + '_'.join(args.modalities), dataset=args.dataset, 
        experiment_id=experiment_id, experiment_config_path=args.experiment_config_path,
        approach=approach)

    # setup callbacks
    callbacks = setup_callbacks(
        early_stopping_metric = "total_val_loss",
        early_stopping_mode   = "min",
        class_names          = dataset_cfg["class_names"],
        num_classes          = len(dataset_cfg["class_names"]),
        no_ckpt              = args.no_ckpt,
        model_weights_path   = args.model_save_path, 
        metric              = 'total_val_loss',
        dataset             = args.dataset, 
        model               = 'mm_' + '_'.join(args.models), 
        experiment_id       = experiment_id,
        include_classification_metrics=use_classification_metrics
    )
    
    for callback in callbacks:
        if isinstance(callback, EarlyStopping):
            callback.monitor = "total_val_loss"
        if isinstance(callback, ModelCheckpoint):
            callback.monitor = "total_val_loss"

    trainer = Trainer.from_argparse_args(
        args=args,
        logger=loggers_list,
        gpus=1,
        precision=16,
        #gradient_clip_val=0.5,
        accumulate_grad_batches=2,
        deterministic=True,
        max_epochs=num_epochs,
        log_every_n_steps=1,
        default_root_dir='logs',
        val_check_interval=0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0,
        callbacks=callbacks,
        enable_checkpointing=not args.no_ckpt
    )

    trainer.fit(model, datamodule)

    if isinstance(model, ContrastiveMultiviewCodingWithDeepAlignment):
        from models.fine_tuning import SupervisedUnimodalHAR
        
        classifier = MultiModalClassifier(
            model.global_encoders,
            out_size=len(dataset_cfg["class_names"]),
            modalities=args.modalities,
            freeze_encoders=True
        )
        
        test_trainer = Trainer(
            accelerator='gpu' if args.gpus else 'cpu',
            devices=1,
            callbacks=[LogClassifierMetrics(
                num_classes=len(dataset_cfg["class_names"]),
                metric_names=['accuracy', 'f1-score', 'precision', 'recall'],
                average='weighted'
            )]
        )
        test_trainer.test(classifier, datamodule)
    else:
        trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}
    
    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics


def setup_callbacks(early_stopping_metric, early_stopping_mode, class_names, num_classes, no_ckpt, model_weights_path, metric, dataset, model, experiment_id, include_classification_metrics=False):
    callbacks = []
    
    early_stop_callback = EarlyStopping(
        monitor='ssl_val_loss',
        min_delta=0.001,
        patience=50,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)

    if not no_ckpt:
        model_checkpoint = ModelCheckpoint(
            monitor='ssl_val_loss',
            dirpath=os.path.join(model_weights_path, dataset, model),
            filename=experiment_id + '-{epoch:02d}-{ssl_val_loss:.2f}',
            save_top_k=1,
            mode='min'
        )
        callbacks.append(model_checkpoint)

    if include_classification_metrics:
        classification_metrics = LogClassifierMetrics(
            num_classes=len(class_names),
            metric_names=['accuracy', 'f1-score', 'precision', 'recall'],
            average='weighted'
        )
        callbacks.append(classification_metrics)

    return callbacks


def setup_classification_metrics(dataset_cfg):
    return LogClassifierMetrics(
        num_classes=len(dataset_cfg["class_names"]),
        metric_names=['accuracy', 'f1-score', 'precision', 'recall'],
        average='weighted'
    )


def main():
    args = parse_arguments()
    cfg = load_yaml_to_dict(args.experiment_config_path)
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]
    seed_everything(cfg['experiment']['seed'])
    train_test_supervised_mm_model(args, cfg, dataset_cfg)


if __name__ == '__main__':
    main()