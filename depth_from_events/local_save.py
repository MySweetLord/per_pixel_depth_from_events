from pathlib import Path
from lightning.pytorch.callbacks import Callback


class LocalCheckpoint(Callback):
    def __init__(self, output_dir, every_n_epochs=5):
        self.output_dir     = Path(output_dir)
        self.every_n_epochs = every_n_epochs
        self.config_yaml    = None

    def set_config(self, config_yaml: str):
        self.config_yaml = config_yaml

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1

        if not trainer.is_global_zero or epoch % self.every_n_epochs != 0:
            return

        if self.config_yaml is None:
            raise RuntimeError("Training config was not provided")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        latest = self.output_dir / "latest.ckpt"
        backup = self.output_dir / "backup.ckpt"

        latest_yaml = latest.with_suffix(".yaml")
        backup_yaml = backup.with_suffix(".yaml")

        # Move previous checkpoint to backup
        if latest.exists():
            latest.replace(backup)

        if latest_yaml.exists():
            latest_yaml.replace(backup_yaml)

        # Save new checkpoint
        trainer.save_checkpoint(str(latest))

        # Save corresponding configuration
        latest_yaml.write_text(self.config_yaml, encoding="utf-8")