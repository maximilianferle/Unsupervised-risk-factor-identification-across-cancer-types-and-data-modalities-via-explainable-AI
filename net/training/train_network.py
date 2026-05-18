from typing import Type

from lightning import Trainer, LightningDataModule
from lightning import seed_everything
from torch.nn import Module
from torchinfo import summary

from .trainable_network import TrainableNetwork


def run_training_session(
        datamodule: Type[LightningDataModule],
        model: Type[Module],
        model_params: dict,
        training_params: dict,
        max_epochs=200,
        seed=0,
        use_cached_dm=False,
):
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
    )

    dm = datamodule(
        batch_size=training_params.pop("batch_size"),
        k=training_params.pop("k"),
        use_cached=use_cached_dm,
    )

    seed_everything(seed=seed)
    trainable_model = TrainableNetwork(
        model=model,
        model_params=model_params,
        **training_params,
    )

    summary(trainable_model.model)

    training_complete = False
    try:
        trainer.fit(trainable_model, datamodule=dm)
        training_complete = True
    except Exception as e:
        print(f"Error during training ({type(e)}): {e}")
    finally:
        results = {
            "val_loss": trainer.callback_metrics["val_loss"].item(),
            "val_entropy": trainer.callback_metrics["val_entropy"].item(),
            "v_num": trainable_model.logger.version,
            "model": trainable_model,
            "dm": dm,
            "training_complete": training_complete,
        }
    return results
