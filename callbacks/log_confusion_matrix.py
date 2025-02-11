import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

class LogConfusionMatrix(pl.Callback):
    """
    A callback which caches all labels and predictions encountered during a testing epoch,
    then logs a confusion matrix to WandB at the end of the test.
    """
    def __init__(self, class_names):
        self.class_names = class_names
        self._reset_state()

    def _reset_state(self):
        self.labels = []
        self.preds = []

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._reset_state()

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx, dataloader_idx) -> None:
        self.labels += (batch['label'] - 1).tolist()
        self.preds += outputs['preds'].tolist()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Retrieve the WandB logger, if it exists.
        wandb_logger = None
        loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                wandb_logger = logger
        if wandb_logger == None:
            return

        # Log the confusion matrix.
        confusion_matrix = wandb.plot.confusion_matrix(
            y_true = self.labels,
            preds = self.preds,
            class_names = self.class_names
        )
        wandb_logger.experiment.log({
            "confusion_matrix": confusion_matrix,
            "global_step": trainer.global_step
        })

        