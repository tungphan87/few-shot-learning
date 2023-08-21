from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
from src.models.model import ProtoNet
from src.data.data_loader import split_train_val_test_sets
from src.data.constants import CHECKPOINT_PATH
import os
import torch

def train_model(model_class, train_loader, val_loader, device, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        enable_progress_bar=False,
    )

    pl.seed_everything(42)
    model = model_class(**kwargs)
    trainer.fit(model, train_loader, val_loader)
    model = model_class.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )

    return model


if __name__ == "__main__":
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    train_data_loader, val_data_loader, test_data_loader = split_train_val_test_sets()
    protonet_model = train_model(
        ProtoNet,
        train_loader=train_data_loader,
        val_loader=val_data_loader,
        proto_dim=64,
        lr=2e-4,
        device=device, 
    )
