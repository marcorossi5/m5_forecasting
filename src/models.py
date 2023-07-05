import pytorch_lightning as pl
import torch
from darts import models

def create_nhits_model(lb, fh):
    batch_size = 2**13
    lr = 1e-3
    optim_kwargs = {
        "lr": lr,
    }
    encoders = {
        "cyclic": {"past": ["dayofweek", "weekofyear"]}
    }
    return models.NHiTSModel(
        input_chunk_length=lb,
        output_chunk_length=fh,
        batch_size=batch_size,
        optimizer_kwargs=optim_kwargs,
        add_encoders=encoders,
    )


def create_nbeats_model(lb, fh):
    batch_size = 2**13
    lr = 1e-3
    optim_kwargs = {
        "lr": lr,
    }
    encoders = {
        "cyclic": {"past": ["dayofweek", "weekofyear"]}
    }
    return models.NBEATSModel(
        input_chunk_length=lb,
        output_chunk_length=fh,
        batch_size=batch_size,
        optimizer_kwargs=optim_kwargs,
        add_encoders=encoders,
    )


def create_tft_model(lb, fh):
    batch_size = 2**10
    lr = 1e-3
    optim_kwargs = {
        "lr": lr,
    }
    encoders = {
        "cyclic": {"past": ["dayofweek", "weekofyear"], "future": ["dayofweek", "weekofyear"]}
    }
    return models.TFTModel(
        input_chunk_length=lb,
        output_chunk_length=fh,
        batch_size=batch_size,
        save_checkpoints=False,
        optimizer_kwargs=optim_kwargs,
        add_encoders=encoders,
    )


def create_model(model_type: str, lb: int, fh: int):
    if model_type == "nhits":
        return create_nhits_model(lb, fh)
    elif model_type == "nbeats":
        return create_nbeats_model(lb, fh)
    elif model_type == "tft":
        return create_tft_model(lb, fh)
    else:
        raise ValueError(f"Model type not recognized, got {model_type}")


def load_model(model_type: str, ckpt_path):
    # device = "gpu" if torch.cuda.is_available() else "cpu"
    if model_type == "nhits":
        return models.NHiTSModel.load(ckpt_path) #, map_location=device)
    elif model_type == "nbeats":
        return models.NBEATSModel.load(ckpt_path) #, map_location=device)
    elif model_type == "tft":
        return models.TFTModel.load(ckpt_path) # , map_location=device)
    else:
        raise ValueError(f"Model type not recognized, got {model_type}")


def get_epochs(model_type: str):
    if model_type == "nhits":
        return 100
    elif model_type == "nbeats":
        return 100
    elif model_type == "tft":
        return 10
    else:
        raise ValueError(f"Model type not recognized, got {model_type}")


def get_trainer(model_type: str):
    """Configure trainer"""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"./training_logs/ckpts/{model_type}",
        save_top_k=5,
        monitor="train_loss",
        mode="min",
    )
    trainer_kwargs = {
        "accelerator": "gpu",
        "default_root_dir": f"./training_logs/logs/{model_type}",
        "enable_checkpointing": True,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "logger": True,
        "log_every_n_steps": 25,
        "precision": 32,
        "max_epochs": get_epochs(model_type),
        "plugins": [pl.plugins.PrecisionPlugin()],
        "callbacks": [checkpoint_callback],
        # "fast_dev_run": True,
    }
    return pl.Trainer(**trainer_kwargs)
