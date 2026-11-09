from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import OmegaConf
import torch


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config):

    # set to prevent warning
    torch.set_float32_matmul_precision("high")

    # reproducibility
    if config.trainer.deterministic:
        seed_everything(42, workers=True)

    # dataset + dataloader = lightning datamodule
    datamodule      = instantiate(config.datamodule)

    # network + transform + loss functions + optimizer = lightning module
    network         = instantiate(config.network)
    transform       = instantiate(config.transform)
    loss_functions  = instantiate(config.loss_functions)
    optimizer       = instantiate(config.optimizer)
    scheduler       = instantiate(config.scheduler)

    litmodule       = instantiate(config.litmodule, network, transform, loss_functions, optimizer, scheduler)

    callbacks       = instantiate(config.callbacks)
    logger          = instantiate(config.logger)

    # trainer and train!
    trainer         = instantiate(config.trainer, logger=logger, callbacks=list(callbacks.values()), enable_checkpointing=True)

    print("start training")
    trainer.fit(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
