from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
import torch


@hydra.main(version_base=None, config_path="config", config_name="validate")  # only hydra config and overrides
def main(overrides):

    torch.set_float32_matmul_precision("high")                                # set to prevent warning


    if overrides.local_load is not None:

        checkpoint = Path(overrides.local_load).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Local checkpoint does not exist: {checkpoint}")

        config_path = checkpoint.with_suffix(".yaml")
        if not config_path.is_file():
            raise FileNotFoundError(f"Training config sidecar does not exist: {config_path}")
        
        config_bytes = config_path.read_bytes()
        encoding     = "utf-16" if config_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        config       = OmegaConf.create(config_bytes.decode(encoding))
    else:
        # Without a local checkpoint, validation stops here; remote W&B loading is no longer available.
        raise ValueError("Provide local_load=<checkpoint.ckpt>; W&B run IDs are no longer supported")


    for key in overrides.deletes:
        config.pop(key, None)
        
    with open_dict(config):
        config.merge_with(overrides)

    # dataset + dataloader = lightning datamodule
    datamodule = instantiate(config.datamodule)

    # network + transform + loss functions = lightning module
    network         = instantiate(config.network)
    transform       = instantiate(config.transform)
    loss_functions  = instantiate(config.loss_functions)
    litmodule       = instantiate(config.litmodule, network, transform, loss_functions, optimizer=None, scheduler=None)

    if overrides.local_state_dict is not None:

        state_dict = torch.load(overrides.local_state_dict, weights_only=True, map_location="cpu")
        new_state_dict = {}
        for key in state_dict:
            new_state_dict[key.replace("_orig_mod", "network")] = state_dict[key]
        state_dict = new_state_dict

    else:

        state_dict = torch.load(checkpoint, weights_only=True, map_location="cpu")["state_dict"]
        if "state_dict_maps" in overrides:  # temporary
            new_state_dict = {}
            for key in state_dict:
                new_key = key
                for before, after in config.state_dict_maps.items():
                    if before in key:
                        new_key = new_key.replace(before, after)
                new_state_dict[new_key] = state_dict[key]
            state_dict = new_state_dict

    litmodule.load_state_dict(state_dict)
    litmodule.eval()
    litmodule.freeze()

    # callbacks
    callbacks = instantiate(config.callbacks)
    callbacks.pop("checkpoint", None)
    callbacks.pop("local_save", None)

    # trainer and validate!
    trainer = instantiate(config.trainer, logger=False, callbacks=[cb for cb in callbacks.values()])
    trainer.validate(litmodule, datamodule=datamodule)


if __name__ == "__main__":
    main()
