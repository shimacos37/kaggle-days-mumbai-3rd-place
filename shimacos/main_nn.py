import hashlib
import os
import random
import re
import shutil
from glob import glob

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import runner


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def prepair_dir(config: DictConfig):
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
        config.store.feature_path,
    ]:
        if (
            os.path.exists(path)
            and config.train.warm_start is False
            and config.data.is_train
        ):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_up(config: DictConfig):
    # Setup
    prepair_dir(config)
    set_seed(config.train.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


@hydra.main(config_path="yamls", config_name="nn.yaml")
def main(config: DictConfig):
    os.chdir(config.store.workdir)
    set_up(config)
    # Preparing for trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.store.model_path,
        filename=config.store.model_name + "-{epoch}-{step}",
        monitor=config.train.callbacks.monitor_metric,
        verbose=True,
        save_top_k=1,
        mode=config.train.callbacks.mode,
        save_weights_only=False,
    )
    hparams = {}
    for key, value in config.items():
        if isinstance(value, DictConfig):
            hparams.update(value)
        else:
            hparams.update({key: value})
    if config.store.wandb_project is not None:
        logger = WandbLogger(
            name=config.store.model_name + f"_fold{config.data.n_fold}",
            save_dir=config.store.log_path,
            project=config.store.wandb_project,
            version=hashlib.sha224(bytes(str(hparams), "utf8")).hexdigest()[:4],
            anonymous=True,
            group=config.store.model_name,
            tags=[config.data.label_col],
        )
    else:
        logger = None

    early_stop_callback = EarlyStopping(
        monitor=config.train.callbacks.monitor_metric,
        patience=config.train.callbacks.patience,
        verbose=True,
        mode=config.train.callbacks.mode,
    )

    backend = "ddp" if len(config.base.gpu_id) > 1 else None
    if config.train.warm_start:
        checkpoint_path = sorted(
            glob(config.store.model_path + "/*epoch*"),
            key=lambda path: int(re.split("[=.]", path)[-2]),
        )[-1]
        print(checkpoint_path)
    else:
        checkpoint_path = None

    model = getattr(runner, config.runner)(config)
    params = {
        "logger": logger,
        "max_epochs": config.train.epoch,
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [early_stop_callback, checkpoint_callback],
        "accumulate_grad_batches": config.train.accumulation_steps,
        # "amp_backend": "native",
        # "amp_level": "",
        "precision": 16,
        "gpus": len(config.base.gpu_id),
        "accelerator": backend,
        # "plugins": "ddp_sharded",
        "limit_train_batches": 1.0,
        "check_val_every_n_epoch": 1,
        "limit_val_batches": 1.0,
        "limit_test_batches": 0.0,
        "num_sanity_val_steps": 5,
        "num_nodes": 1,
        "gradient_clip_val": 0.5,
        "log_every_n_steps": 10,
        "flush_logs_every_n_steps": 10,
        "profiler": "simple",
        "deterministic": False,
        "resume_from_checkpoint": checkpoint_path,
        "weights_summary": "top",
        "reload_dataloaders_every_epoch": True,
        "replace_sampler_ddp": False,
    }
    if config.data.is_train:
        trainer = Trainer(**params)
        trainer.fit(model)
    else:
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model.load_state_dict(state_dict)
        params.update(
            {
                "gpus": 1,
                "logger": None,
                "limit_train_batches": 0.0,
                "limit_val_batches": 0.0,
                "limit_test_batches": 1.0,
                "accelerator": None,
            }
        )
        trainer = Trainer(**params)
        trainer.test(model, model.test_dataloader())


if __name__ == "__main__":
    main()
