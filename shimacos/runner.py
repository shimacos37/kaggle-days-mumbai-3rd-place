import io
import multiprocessing as mp
import os
import yaml
from glob import glob
from typing import Dict, List, Tuple, Union
from itertools import chain
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import wandb
import matplotlib.pyplot as plt
from google.cloud import storage
from omegaconf import DictConfig
from pytorch_lightning.utilities.distributed import rank_zero_info
from sklearn.metrics import mean_squared_log_error, f1_score
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

plt.style.use("seaborn-whitegrid")

from src.factories import (
    get_dataset,
    get_loss,
    get_model,
    get_optimizer,
    get_sampler,
    get_scheduler,
)
from src.sync_batchnorm import convert_model

try:
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass


def text_collate(batch):
    output = {}
    max_length = max([sample["raw_length"] for sample in batch])
    output["id"] = [sample["id"] for sample in batch]
    output["input_ids"] = torch.stack(
        [torch.as_tensor(sample["input_ids"][:max_length]) for sample in batch],
    )
    output["attention_mask"] = torch.stack(
        [torch.as_tensor(sample["attention_mask"][:max_length]) for sample in batch],
    )
    output["feature"] = torch.stack(
        [torch.as_tensor(sample["feature"]) for sample in batch],
    )
    output["lang_code"] = torch.stack(
        [torch.as_tensor(sample["lang_code"]) for sample in batch],
    )
    output["raw_length"] = [sample["raw_length"] for sample in batch]
    if "category" in batch[0]:
        output["category"] = torch.cat(
            [torch.as_tensor([sample["category"]]) for sample in batch],
            dim=0,
        )
    return output


class BaseRunner(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super(BaseRunner, self).__init__()
        self.base_config = hparams.base
        self.data_config = hparams.data
        self.model_config = hparams.model
        self.train_config = hparams.train
        self.test_config = hparams.test
        self.store_config = hparams.store
        # load from factories
        if self.data_config.is_train:
            self.train_dataset = get_dataset(config=self.data_config, mode="train")
            self.valid_dataset = get_dataset(config=self.data_config, mode="valid")
            if self.data_config.n_fold != "all":
                self.num_train_optimization_steps = int(
                    self.train_config.epoch
                    * len(self.train_dataset)
                    / (self.train_config.batch_size)
                    / self.train_config.accumulation_steps
                    / len(self.base_config.gpu_id)
                )
            else:
                self.num_train_optimization_steps = int(
                    50
                    * len(self.train_dataset)
                    / (self.train_config.batch_size)
                    / self.train_config.accumulation_steps
                    / len(self.base_config.gpu_id)
                )
            if hparams.debug:
                print(self.train_dataset.__getitem__(0))
        else:
            if self.test_config.is_validation:
                self.test_dataset = get_dataset(config=self.data_config, mode="valid")
                self.prefix = "valid"
            else:
                self.test_dataset = get_dataset(config=self.data_config, mode="test")
                self.prefix = "test"
            self.num_train_optimization_steps = 100
            if hparams.debug:
                print(self.test_dataset.__getitem__(0))
        self.model = get_model(self.model_config)
        self.loss = get_loss(loss_class=self.base_config.loss_class)
        # path setting
        self.initialize_variables()
        self.save_flg = False
        self.refinement_step = False

        if len(self.base_config.gpu_id) > 1:
            self.model = convert_model(self.model)
        self.cpu_count = mp.cpu_count() // len(self.base_config.gpu_id)

        # column
        self.id_col = self.data_config.id_col
        self.label_col = self.data_config.label_col
        self.pred_col = self.data_config.pred_col

    def configure_optimizers(self):
        if self.base_config.use_transformer_parameter:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias"]
            optimizer_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.001,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_parameters = self.model.parameters()

        optimizer = get_optimizer(
            opt_class=self.base_config.opt_class,
            params=optimizer_parameters,
            lr=self.train_config.learning_rate,
        )
        if self.base_config.scheduler_class == "GradualWarmupScheduler":
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.num_train_optimization_steps, eta_min=1e-6
            )
            scheduler = {
                "scheduler": get_scheduler(
                    scheduler_class=self.base_config.scheduler_class,
                    optimizer=optimizer,
                    multiplier=10,
                    total_epoch=int(self.num_train_optimization_steps * 0.01),
                    after_scheduler=scheduler_cosine,
                ),
                "interval": "step",
            }
            scheduler["scheduler"].step(self.step)
        elif self.base_config.scheduler_class == "ReduceLROnPlateau":
            scheduler = {
                "scheduler": get_scheduler(
                    scheduler_class=self.base_config.scheduler_class,
                    optimizer=optimizer,
                    mode=self.train_config.callbacks.mode,
                    factor=0.5,
                    patience=self.train_config.scheduler.patience,
                    verbose=True,
                ),
                "interval": "epoch",
                "monitor": self.train_config.callbacks.monitor_metric,
            }
        else:
            raise NotImplementedError
        return [optimizer], [scheduler]

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.model(batch)

    def train_dataloader(self):
        if self.trainer.num_gpus > 1:
            if self.data_config.dataset_class == "query_dataset":
                sampler = get_sampler("weighted_sampler", dataset=self.train_dataset)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                )
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self.cpu_count,
            sampler=sampler,
            collate_fn=text_collate,
        )
        return train_loader

    def val_dataloader(self):
        if self.trainer.num_gpus > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(
                self.valid_dataset,
            )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.cpu_count,
            pin_memory=True,
            sampler=sampler,
            collate_fn=text_collate,
        )

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.cpu_count,
            collate_fn=text_collate,
        )
        return test_loader

    def initialize_variables(self):
        self.step = 0
        if self.train_config.callbacks.mode == "max":
            self.best_score = -np.inf
            self.moninor_op = np.greater_equal
        elif self.train_config.callbacks.mode == "min":
            self.best_score = np.inf
            self.moninor_op = np.less_equal
        if self.train_config.warm_start:
            pass
            # with open(f"{self.store_config.log_path}/best_score.yaml", "r") as f:
            #     res = yaml.safe_load(f)
            # if "best_score" in res.keys():
            # self.best_score = res["best_score"]
            # self.step = res["step"]

    def upload_directory(self, path):
        storage_client = storage.Client(self.store_config.gcs_project)
        bucket = storage_client.get_bucket(self.store_config.bucket_name)
        filenames = glob(f"{path}/**", recursive=True)
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            destination_blob_name = f"{self.store_config.gcs_path}/{filename.split(self.store_config.save_path)[-1][1:]}"
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(filename)

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = (
            running_train_loss.cpu().item()
            if running_train_loss is not None
            else float("NaN")
        )
        tqdm_dict = {"loss": "{:2.6g}".format(avg_training_loss)}

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict["v_num"] = version

        return tqdm_dict


class ClassificationRunner(BaseRunner):
    def __init__(self, hparams: DictConfig):
        super(ClassificationRunner, self).__init__(hparams)

    def training_step(self, batch, batch_nb):

        pred = self.forward(batch)
        loss = self.loss(pred, batch[self.label_col])
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "lr",
            self.trainer.lr_schedulers[0]["scheduler"].optimizer.param_groups[0]["lr"],
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_nb):
        pred = self.forward(batch)
        if self.test_config.is_tta:
            pred = pred.reshape(3, pred.shape[0] // 3, pred.shape[-1]).mean(0)
        label = batch[self.label_col]
        loss = self.loss(pred, label)
        pred = F.softmax(pred, dim=-1)

        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        metrics = {
            self.id_col: batch[self.id_col],
            self.pred_col: pred,
            self.label_col: label,
            "loss": loss,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature()
            metrics.update({"feature": feature.detach().cpu().numpy()})

        return metrics

    def test_step(self, batch, batch_nb):
        if self.test_config.is_tta:
            batch["data"] = torch.cat(
                [batch["data"], batch["data"].flip(2), batch["data"].flip(1)], dim=0
            )  # horizontal flip
        with torch.no_grad():
            pred = self.forward(batch)
        if self.test_config.is_tta:
            pred = pred.reshape(3, pred.shape[0] // 3, pred.shape[-1]).mean(0)
        pred = F.softmax(pred, dim=-1)
        pred = pred.detach().cpu().numpy()
        metrics = {
            self.id_col: batch[self.id_col],
            self.pred_col: pred,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature().detach().cpu().numpy()
            metrics.update({"feature": feature})
        if self.test_config.is_validation:
            label = batch[self.label_col].detach().cpu().numpy()
            metrics.update({self.label_col: label})
        return metrics

    def _compose_result(self, outputs: List[Dict[str, np.ndarray]]) -> pd.DataFrame:

        preds = np.concatenate([x[self.pred_col] for x in outputs], axis=0)
        ids = np.concatenate([x[self.id_col] for x in outputs]).reshape(
            -1,
        )
        if self.data_config.is_train:
            dataset = self.valid_dataset
        else:
            dataset = self.test_dataset
        df_dict = {self.id_col: ids}
        if len(preds.shape) > 1:
            for i in range(preds.shape[1]):
                df_dict[f"{self.pred_col}_{i}"] = preds[:, i]
            df_dict[self.pred_col] = np.argmax(preds, axis=1)
        else:
            df_dict[self.pred_col] = preds
        if self.label_col in outputs[0].keys():
            label = np.concatenate([x[self.label_col] for x in outputs])
            if len(label.shape) > 1:
                df_dict[self.label_col] = list(label)
            else:
                df_dict[self.label_col] = label
        df = pd.DataFrame(df_dict)
        return df

    def _calc_metric(self, label: np.ndarray, pred: np.ndarray) -> float:
        return f1_score(label, pred, average="micro")

    def validation_epoch_end(self, outputs: List[Dict[str, np.ndarray]]):
        loss = np.mean([x["loss"].item() for x in outputs])
        if self.store_config.save_feature:
            features = np.concatenate([x["feature"] for x in outputs], axis=0)
        df = self._compose_result(outputs)
        if self.trainer.num_gpus > 1:
            # DDP使用時の結果をまとめる処理
            rank = dist.get_rank()
            df.to_csv(f"{self.store_config.result_path}/valid_{rank}.csv", index=False)
            dist.barrier()
            metrics = {"avg_loss": loss}
            world_size = dist.get_world_size()
            aggregated_metrics = {}
            for metric_name, metric_val in metrics.items():
                metric_tensor = torch.tensor(metric_val).to(f"cuda:{rank}")
                dist.barrier()
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                reduced_metric = metric_tensor.item() / world_size
                aggregated_metrics[metric_name] = reduced_metric
            loss = aggregated_metrics["avg_loss"]
        else:
            pass
        if self.trainer.num_gpus > 1:
            # 各rankでlocalに保存した結果のcsvをまとめる
            paths = sorted(glob(f"{self.store_config.result_path}/valid_[0-9].csv"))
            df = pd.concat([pd.read_csv(path) for path in paths]).reset_index(drop=True)
        # NNの評価指標
        f1 = self._calc_metric(df[self.label_col], df[self.pred_col])
        metrics = {}
        metrics["val_loss"] = float(loss)
        metrics["f1"] = float(f1)

        # scoreが改善した時のみ結果を保存
        # df.shape[0] >= 2000はsanity_stepの時はskipさせるため
        if self.moninor_op(
            metrics[self.train_config.callbacks.monitor_metric], self.best_score
        ):
            self.best_score = metrics[self.train_config.callbacks.monitor_metric]
            self.save_flg = True
            res = {}
            res["step"] = int(self.global_step)
            res["epoch"] = int(self.current_epoch)
            res["best_score"] = self.best_score
            df.to_csv(f"{self.store_config.result_path}/valid.csv", index=False)
            with open(f"{self.store_config.log_path}/best_score.yaml", "w") as f:
                yaml.dump(res, f, default_flow_style=False)
        for key, val in metrics.items():
            self.log(key, val, prog_bar=True)
        self.log("best_score", self.best_score, prog_bar=True)

    def on_epoch_end(self):
        def _tile_plot(n_ids=100, n_cols=10):
            n_rows = n_ids // n_cols + 1
            fig = plt.figure(figsize=(int(n_cols * 2.3), int(n_rows * 1.5)))
            for i, id_ in enumerate(df[self.id_col].unique()[:n_ids]):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                tmp_df = df.query(f"{self.id_col}==@id_")
                ax.plot(
                    tmp_df["time_step"].iloc[0],
                    tmp_df[self.label_col].iloc[0],
                    label=self.label_col,
                    alpha=0.5,
                )
                ax.plot(
                    tmp_df["time_step"].iloc[0],
                    tmp_df[self.pred_col].iloc[0],
                    label=self.pred_col,
                    alpha=0.5,
                )
                ax.set_title(f"{self.id_col}={id_}")
                ax.legend()
            fig.tight_layout()
            return fig

        if self.save_flg:
            if self.store_config.gcs_project is not None:
                # self.upload_directory()
                pass
            if self.trainer.logger is not None:
                # df = pd.read_csv(f"{self.store_config.result_path}/valid.csv")
                # fig = _tile_plot()
                # self.trainer.logger.experiment.log(
                #     {"prediction": wandb.Image(fig)}, step=self.global_step
                # )
                pass
            self.save_flg = False
        if self.current_epoch >= self.train_config.refinement_step:
            self.train_dataset.set_refinement_step()

    def test_epoch_end(self, outputs):
        df = self._compose_result(outputs)
        if self.store_config.save_feature:
            features = np.concatenate([x["feature"] for x in outputs], axis=0)
        if self.trainer.num_gpus > 1:
            rank = dist.get_rank()
            df.to_csv(
                f"{self.store_config.result_path}/{self.prefix}_{rank}.csv", index=False
            )
            dist.barrier()
            paths = sorted(
                glob(f"{self.store_config.result_path}/{self.prefix}_[0-9].csv")
            )
            df = pd.concat([pd.read_csv(path) for path in paths])
        df.to_csv(f"{self.store_config.result_path}/{self.prefix}.csv", index=False)
        if not self.test_config.is_validation:
            # testデータに対する推論
            idx2label = {val: key for key, val in self.data_config.label_map.items()}
            if self.data_config.n_fold == 4:
                test_df = pd.read_csv("./input/sample_submission.csv")
                sub_dict = {}
                dfs = [
                    pd.read_csv(
                        f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/test.csv"
                    )
                    for i in range(5)
                ]
                probs = []
                preds = []
                prob_cols = [
                    f"{self.pred_col}_{i}" for i in range(self.model_config.num_classes)
                ]
                for df in dfs:
                    probs.append(df[prob_cols].values)
                sub_dict[self.label_col] = np.mean(probs, axis=0).argmax(1)
                test_df[self.label_col] = np.mean(probs, axis=0).argmax(1)
                for i in range(self.model_config.num_classes):
                    test_df[f"{self.pred_col}_{i}"] = np.mean(probs, axis=0)[:, i]
                sub_dict[self.id_col] = test_df[self.id_col]
                sub = pd.DataFrame(sub_dict)
                sub[self.label_col] = sub[self.label_col].map(idx2label)
                test_df.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/test.csv",
                    index=False,
                )
                sub.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/submission.csv",
                    index=False,
                )
                self.upload_directory(self.store_config.root_path)
        else:
            result = {}
            # validationデータに対する推論
            if self.data_config.n_fold == 4:
                dfs = pd.concat(
                    [
                        pd.read_csv(
                            f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/valid.csv"
                        )
                        for i in range(5)
                    ],
                    axis=0,
                )
                dfs.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/{self.store_config.model_name}.csv",
                )
                self.upload_directory(self.store_config.root_path)
                result.update({"score_all": dfs["f1"].mean()})
                return {"score_all": dfs["f1"].mean()}

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """
        self.upload_directory(self.store_config.save_path)
        if self.data_config.n_fold == 4 and self.data_config.is_train:
            dfs = pd.concat(
                [
                    pd.read_csv(
                        f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/valid.csv"
                    )
                    for i in range(5)
                ],
                axis=0,
            )
            # score = dfs["f1"].mean()
            # rank_zero_info(f"all_score: {score}")
            dfs.to_csv(
                f"{self.store_config.root_path}/{self.store_config.model_name}/valid.csv",
                index=False,
            )
            self.upload_directory(self.store_config.root_path)


class RegressionRunner(BaseRunner):
    def __init__(self, hparams: DictConfig):
        super(RegressionRunner, self).__init__(hparams)

    def training_step(self, batch, batch_nb):
        pred = self.forward(batch)
        loss = self.loss(pred.double(), batch[self.label_col])
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "lr",
            self.trainer.lr_schedulers[0]["scheduler"].optimizer.param_groups[0]["lr"],
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_nb):
        pred = self.forward(batch)
        label = batch[self.label_col]
        loss = self.loss(pred, label)
        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        metrics = {
            self.id_col: batch[self.id_col],
            self.pred_col: pred,
            self.label_col: label,
            "loss": loss,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature()
            metrics.update({"feature": feature.detach().cpu().numpy()})

        return metrics

    def test_step(self, batch, batch_nb):
        pred = self.forward(batch)
        pred = pred.detach().cpu().numpy()
        metrics = {
            self.id_col: batch[self.id_col],
            self.pred_col: pred,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature().detach().cpu().numpy()
            metrics.update({"feature": feature})
        if self.test_config.is_validation:
            label = batch[self.label_col].detach().cpu().numpy()
            metrics.update({self.label_col: label})
        return metrics

    def _compose_result(self, outputs: List[Dict[str, np.ndarray]]) -> pd.DataFrame:

        preds = np.concatenate([x[self.pred_col] for x in outputs], axis=0)
        ids = np.concatenate([x[self.id_col] for x in outputs]).reshape(
            -1,
        )
        if self.data_config.is_train:
            dataset = self.valid_dataset
        else:
            dataset = self.test_dataset
        df_dict = {self.id_col: ids}
        if len(preds.shape) > 1:
            for i in range(preds.shape[1]):
                df_dict[f"{self.pred_col}_{i}"] = preds[:, i]
            df_dict[self.pred_col] = np.argmax(preds, axis=1)
        else:
            df_dict[self.pred_col] = preds
        if self.label_col in outputs[0].keys():
            label = np.concatenate([x[self.label_col] for x in outputs])
            if len(label.shape) > 1:
                df_dict[self.label_col] = list(label)
            else:
                df_dict[self.label_col] = label
        df = pd.DataFrame(df_dict)
        return df

    def _calc_metric(self, label: np.ndarray, pred: np.ndarray) -> float:
        return f1_score(label, pred, average="macro")

    def validation_epoch_end(self, outputs: List[Dict[str, np.ndarray]]):
        loss = np.mean([x["loss"].item() for x in outputs])
        if self.store_config.save_feature:
            features = np.concatenate([x["feature"] for x in outputs], axis=0)
        df = self._compose_result(outputs)
        if self.trainer.num_gpus > 1:
            # DDP使用時の結果をまとめる処理
            rank = dist.get_rank()
            df.to_csv(f"{self.store_config.result_path}/valid_{rank}.csv", index=False)
            dist.barrier()
            metrics = {"avg_loss": loss}
            world_size = dist.get_world_size()
            aggregated_metrics = {}
            for metric_name, metric_val in metrics.items():
                metric_tensor = torch.tensor(metric_val).to(f"cuda:{rank}")
                dist.barrier()
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                reduced_metric = metric_tensor.item() / world_size
                aggregated_metrics[metric_name] = reduced_metric
            loss = aggregated_metrics["avg_loss"]
        else:
            pass
        if self.trainer.num_gpus > 1:
            # 各rankでlocalに保存した結果のcsvをまとめる
            paths = sorted(glob(f"{self.store_config.result_path}/valid_[0-9].csv"))
            df = pd.concat([pd.read_csv(path) for path in paths]).reset_index(drop=True)
        # NNの評価指標
        df[self.label_col] = np.expm1(df[self.label_col])
        df[self.pred_col] = np.clip(np.expm1(df[self.pred_col]), 0, 1)
        rmsle = np.sqrt(mean_squared_log_error(df[self.label_col], df[self.pred_col]))
        metrics = {}
        metrics["val_loss"] = float(loss)
        metrics["rmsle"] = rmsle

        # scoreが改善した時のみ結果を保存
        # df.shape[0] >= 2000はsanity_stepの時はskipさせるため
        if self.moninor_op(
            metrics[self.train_config.callbacks.monitor_metric], self.best_score
        ):
            self.best_score = metrics[self.train_config.callbacks.monitor_metric]
            self.save_flg = True
            res = {}
            res["step"] = int(self.global_step)
            res["epoch"] = int(self.current_epoch)
            res["best_score"] = self.best_score
            df.to_csv(f"{self.store_config.result_path}/valid.csv", index=False)
            with open(f"{self.store_config.log_path}/best_score.yaml", "w") as f:
                yaml.dump(res, f, default_flow_style=False)
        for key, val in metrics.items():
            self.log(key, val, prog_bar=True)
        self.log("best_score", self.best_score, prog_bar=True)

    def on_epoch_end(self):
        def _tile_plot(n_ids=100, n_cols=10):
            n_rows = n_ids // n_cols + 1
            fig = plt.figure(figsize=(int(n_cols * 2.3), int(n_rows * 1.5)))
            for i, id_ in enumerate(df[self.id_col].unique()[:n_ids]):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                tmp_df = df.query(f"{self.id_col}==@id_")
                ax.plot(
                    tmp_df["time_step"].iloc[0],
                    tmp_df[self.label_col].iloc[0],
                    label=self.label_col,
                    alpha=0.5,
                )
                ax.plot(
                    tmp_df["time_step"].iloc[0],
                    tmp_df[self.pred_col].iloc[0],
                    label=self.pred_col,
                    alpha=0.5,
                )
                ax.set_title(f"{self.id_col}={id_}")
                ax.legend()
            fig.tight_layout()
            return fig

        if self.save_flg:
            if self.store_config.gcs_project is not None:
                # self.upload_directory()
                pass
            if self.trainer.logger is not None:
                # df = pd.read_csv(f"{self.store_config.result_path}/valid.csv")
                # fig = _tile_plot()
                # self.trainer.logger.experiment.log(
                #     {"prediction": wandb.Image(fig)}, step=self.global_step
                # )
                pass
            self.save_flg = False
        if self.current_epoch >= self.train_config.refinement_step:
            self.train_dataset.set_refinement_step()

    def test_epoch_end(self, outputs):
        df = self._compose_result(outputs)
        if self.store_config.save_feature:
            features = np.concatenate([x["feature"] for x in outputs], axis=0)
        if self.trainer.num_gpus > 1:
            rank = dist.get_rank()
            df.to_csv(
                f"{self.store_config.result_path}/{self.prefix}_{rank}.csv", index=False
            )
            dist.barrier()
            paths = sorted(
                glob(f"{self.store_config.result_path}/{self.prefix}_[0-9].csv")
            )
            df = pd.concat([pd.read_csv(path) for path in paths])
        df.to_csv(f"{self.store_config.result_path}/{self.prefix}.csv", index=False)
        if not self.test_config.is_validation:
            # testデータに対する推論
            if self.data_config.n_fold == 3:
                test_df = pd.read_csv("./input/sample_submission.csv")
                sub_dict = {}
                dfs = [
                    pd.read_csv(
                        f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/test.csv"
                    )
                    for i in range(4)
                ]
                preds = []
                for df in dfs:
                    preds.append(df[self.pred_col])
                sub_dict[self.label_col] = np.clip(
                    np.expm1(np.mean(preds, axis=0)), 0, 1
                )
                test_df[self.label_col] = np.clip(
                    np.expm1(np.mean(preds, axis=0)), 0, 1
                )
                sub_dict[self.id_col] = df["id"]
                test_df.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/test.csv",
                    index=False,
                )
                pd.DataFrame(sub_dict).to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/submission.csv",
                    index=False,
                )
                self.upload_directory(self.store_config.root_path)
        else:
            result = {}
            # validationデータに対する推論
            if self.data_config.n_fold == 3:
                dfs = pd.concat(
                    [
                        pd.read_csv(
                            f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/valid.csv"
                        )
                        for i in range(4)
                    ],
                    axis=0,
                )
                dfs.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}/{self.store_config.model_name}.csv",
                )
                self.upload_directory(self.store_config.root_path)
                result.update({"score_all": dfs["f1"].mean()})
                return {"score_all": dfs["f1"].mean()}

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """
        self.upload_directory(self.store_config.save_path)
        if self.data_config.n_fold == 3 and self.data_config.is_train:
            dfs = pd.concat(
                [
                    pd.read_csv(
                        f"{self.store_config.root_path}/{self.store_config.model_name}/fold{i}/result/valid.csv"
                    )
                    for i in range(4)
                ],
                axis=0,
            )
            # score = dfs["f1"].mean()
            # rank_zero_info(f"all_score: {score}")
            dfs.to_csv(
                f"{self.store_config.root_path}/{self.store_config.model_name}/valid.csv",
                index=False,
            )
            print(
                np.sqrt(mean_squared_log_error(dfs[self.label_col], dfs[self.pred_col]))
            )
            self.upload_directory(self.store_config.root_path)
