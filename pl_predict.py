import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import hydra
from models.three_d.pl_models import BaseModel
from models.three_d.model_selector import select_model
from monai_dataloader import MhdDataset
import sys
import torch
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

torch.set_float32_matmul_precision("medium")


@hydra.main(config_path="./conf", config_name="config", version_base="1.3")
def main(config):
    config = config["config"]
    model_name = config.name
    trainer = L.Trainer(
        **config.trainer,
        default_root_dir=config.hydra_path,
    )

    dataset = MhdDataset(**config.data, is_train=False, fast_mode=True)
    dataloader = dataset.dataloader
    model = select_model(
        model_name, model_params=config.model, pl_model_params=config.pl_models
    )
    model.eval()
    assert config.ckpt is not None, "ckpt is not provided"
    model.init_from_ckpt(config.ckpt)
    trainer.test(model, dataloaders=dataloader)


if __name__ == "__main__":
    main()
