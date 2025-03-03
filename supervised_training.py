import argparse
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from utils.experiment_utils import generate_experiment_id, load_yaml_to_dict
from utils.training_utils import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    
    # Configs
    parser.add_argument('--experiment_config_path', required=True)
    parser.add_argument('--dataset_config_path', default='configs/dataset_configs.yaml')
    parser.add_argument('--tuning_config_path', default=None)

    # Data and models
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--protocol', default='cross_subject')
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_save_path', default='./model_weights')

    parser.add_argument('--no_ckpt', action='store_true', default=False)
    parser.add_argument('--sweep', action='store_true', help='Enable sweep mode')
    
    return parser.parse_args()


class SupervisedModel(LightningModule):
    def __init__(self, encoder, num_classes, modality, lr=0.001, optimizer_name="adam", metric_name="accuracy"):
        super().__init__()
        self.save_hyperparameters('modality', 'lr', 'optimizer_name', 'metric_name')
        self.encoder = encoder
        self.metric_name = metric_name
        self.modality = modality
        
        if metric_name == "accuracy":
            self.train_metric = Accuracy()
            self.val_metric = Accuracy()
            self.test_metric = Accuracy()
        elif metric_name == "f1-score":
            self.train_metric = F1Score(num_classes=num_classes, average='macro')
            self.val_metric = F1Score(num_classes=num_classes, average='macro')
            self.test_metric = F1Score(num_classes=num_classes, average='macro')

    def forward(self, x):
        # Handle both standard encoders and IMUFormer/PoseFormer
        out = self.encoder(x)
        if isinstance(out, tuple):
            # IMUFormer/PoseFormer return (global_features, local_features)
            return out[0]  # Use global features
        return out

    def training_step(self, batch, batch_idx):
        # Handle batch format from datamodule
        x = batch['inertial' if self.modality == 'inertial' else 'skeleton']
        y = batch['label']
        # Shift labels to be 0-based
        y = y - 1
        
        # Debug info
        # print(f"\nInput shape: {x.shape}")
        # print(f"Label shape: {y.shape}")
        # print(f"Label values: min={y.min()}, max={y.max()}")
        
        out = self(x)
        # print(f"Output shape: {out.shape}")
        
        loss = F.cross_entropy(out, y)
        self.train_metric(out.softmax(dim=-1), y)
        self.log(f'train_{self.metric_name}', self.train_metric, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['inertial' if self.modality == 'inertial' else 'skeleton']
        y = batch['label']
        # Shift labels to be 0-based
        y = y - 1
        
        # Debug info
        # print(f"\nVal input shape: {x.shape}")
        # print(f"Val label shape: {y.shape}")
        # print(f"Val label values: min={y.min()}, max={y.max()}")
        
        out = self(x)
        # print(f"Val output shape: {out.shape}")
        
        loss = F.cross_entropy(out, y)
        self.val_metric(out.softmax(dim=-1), y)
        self.log(f'val_{self.metric_name}', self.val_metric, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        # Return predictions and labels for the callback
        return {
            'preds': torch.argmax(out, dim=-1),
            'labels': y,
            'loss': loss
        }

    def test_step(self, batch, batch_idx):
        x = batch['inertial' if self.modality == 'inertial' else 'skeleton']
        y = batch['label']
        # Shift labels to be 0-based
        y = y - 1
        out = self(x)
        self.test_metric(out.softmax(dim=-1), y)
        self.log(f'test_{self.metric_name}', self.test_metric)
        
        # Return predictions and labels for the callback
        return {
            'preds': torch.argmax(out, dim=-1),
            'labels': y
        }

    def configure_optimizers(self):
        if self.hparams.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=getattr(self.hparams, 'weight_decay', 0.0)  # Get weight_decay from hparams
            )
            
            # Use ReduceLROnPlateau for validation loss monitoring
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=20
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss',
                    "frequency": 1
                }
            }
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer_name} not supported")


def train_test_supervised_model(args, cfg, dataset_cfg, freeze_encoder=False, approach='supervised', experiment_info=None, limited_k=None):
    experiment_id = generate_experiment_id()

    modality = list(cfg['modalities'].keys())[0] # assume unimodal for now
    batch_size = cfg['modalities'][modality]['model'][args.model]['kwargs']['batch_size']
    num_epochs = cfg['experiment']['num_epochs']

    model_cfg = cfg['modalities'][modality]['model'][args.model]
    transform_cfg = cfg['modalities'][modality]['transforms']
    model_cfg, transform_cfg = check_sampling_cfg(model_cfg, transform_cfg)
    train_transforms, test_transforms = init_transforms(modality, transform_cfg)
    datamodule = init_datamodule(data_path=args.data_path, dataset_name=args.dataset, modalities=[modality], batch_size=batch_size,
        split=dataset_cfg['protocols'][args.protocol], train_transforms=train_transforms, test_transforms=test_transforms,
        limited_k=limited_k)

    # Merge general model params with dataset-specific model params.
    model_cfg['kwargs'] = {**dataset_cfg[modality], **model_cfg['kwargs']}
    encoder = init_model(model_cfg, dataset_cfg['main_metric'])

    if freeze_encoder:
        encoder.freeze()

    # Create supervised model wrapper
    # print(f"\nDataset info:")
    # print(f"Number of classes: {dataset_cfg['n_classes']}")
    # print(f"Class names: {dataset_cfg['class_names']}")
    
    model = SupervisedModel(
        encoder=encoder,
        num_classes=dataset_cfg['n_classes'],
        modality=modality,
        lr=model_cfg['kwargs'].get('lr', 0.001),
        optimizer_name=model_cfg['kwargs'].get('optimizer_name', 'adam'),
        metric_name=dataset_cfg['main_metric']
    )

    if experiment_info is None:
        experiment_info = {
            "dataset": args.dataset,
            "model": model_cfg['class_name']
        }

    callbacks = setup_callbacks(
        early_stopping_metric = "val_accuracy",
        early_stopping_mode   = "max",
        class_names           = dataset_cfg["class_names"],
        num_classes           = len(dataset_cfg["class_names"]),
        no_ckpt               = args.no_ckpt,
        model_weights_path    = args.model_save_path, 
        metric                = 'val_' + dataset_cfg['main_metric'], 
        dataset               = args.dataset, 
        model                 = args.model, 
        experiment_id         = experiment_id
    )
    # setup loggers: tensorboards and/or wandb with correct entity
    loggers_list, loggers_dict = setup_loggers(
        tb_dir="tb_logs", 
        experiment_info=experiment_info, 
        modality=modality, 
        dataset=args.dataset, 
        experiment_id=experiment_id, 
        experiment_config_path=args.experiment_config_path,
        entity='fabiang',
        approach=approach
    )

    trainer = Trainer.from_argparse_args(
        args=args,
        logger=loggers_list,
        gpus=1,
        deterministic=True,
        max_epochs=num_epochs,
        default_root_dir='logs',
        log_every_n_steps=1,
        val_check_interval = 0.0 if 'val' not in dataset_cfg['protocols'][args.protocol] else 1.0,
        callbacks=callbacks,
        checkpoint_callback=not args.no_ckpt
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule, ckpt_path='best')

    metrics = {metric: float(val) for metric, val in trainer.callback_metrics.items()}

    if 'wandb' in loggers_dict:
        loggers_dict['wandb'].experiment.finish()

    return metrics


def main():
    args = parse_arguments()
    cfg = load_yaml_to_dict(args.experiment_config_path)
    seed_everything(cfg['experiment']['seed'])
    dataset_cfg = load_yaml_to_dict(args.dataset_config_path)['datasets'][args.dataset]
    modality = list(cfg['modalities'].keys())[0] 
    if args.tuning_config_path is None:
        train_test_supervised_model(args, cfg, dataset_cfg)
    else:
        tuning_cfg_combinations = get_tuning_grid_list(args.tuning_config_path, modality, args.model)
        for combination in tuning_cfg_combinations:
            print(combination)
            cfg['modalities'][modality]['model'][args.model]['kwargs'] = {**cfg['modalities'][modality]['model'][args.model]['kwargs'], **combination}
            train_test_supervised_model(args, cfg, dataset_cfg)


if __name__ == '__main__':
    main()
