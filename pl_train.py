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
    callbacks = ModelCheckpoint(
        monitor="train/dice",
        dirpath=config.hydra_path,
        filename="train-epoch{epoch:02d}-dice{train/dice:.2f}",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        **config.trainer,
        callbacks=[callbacks],
        default_root_dir=config.hydra_path,
    )

    dataset = MhdDataset(**config.data, is_train=True, fast_mode=False)
    dataloader = dataset.dataloader
    model = select_model(
        model_name, model_params=config.model, pl_model_params=config.pl_models
    )
    if config.ckpt:
        model.init_from_ckpt(config.ckpt)
    trainer.fit(model, train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
