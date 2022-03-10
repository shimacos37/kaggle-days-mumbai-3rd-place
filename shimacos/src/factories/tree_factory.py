import logging
import os
import pickle
from datetime import timedelta
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.metrics import average_precision_score

import catboost as cat

plt.style.use("seaborn-whitegrid")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "[%(asctime)s] [%(name)s] [L%(lineno)d] [%(levelname)s][%(funcName)s] %(message)s "
    )
)
logger.addHandler(handler)


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するためのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, lgb.Booster] = {}

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def _custom_objective(self, preds: np.ndarray, data: lgb.Dataset):
        labels = data.get_label()
        weight = data.get_weight()
        grad = 2 * weight * (preds - labels)
        hess = 2 * weight
        return grad, hess

    def _custom_metric(self, y_pred: np.ndarray, dtrain: lgb.basic.Dataset):
        y_true = dtrain.get_label().astype(float)
        score = average_precision_score(y_true, y_pred)
        return "average_precision_score", score, True

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(self.config.n_fold):
            bst = self.fit(train_df, n_fold)
            valid_df = train_df.query("fold == @n_fold")
            if self.config.params.objective == "multiclass":
                preds = bst.predict(valid_df[self.config.feature_cols])
                test_preds = bst.predict(test_df[self.config.feature_cols])
                for i in range(self.config.params.num_class):
                    train_df.loc[
                        valid_df.index, f"{self.config.label_col}_prob{i}"
                    ] = preds[:, i]
                    test_df[
                        f"{self.config.label_col}_prob{i}_fold{n_fold}"
                    ] = test_preds[:, i]
            else:
                train_df.loc[valid_df.index, self.config.pred_col] = bst.predict(
                    valid_df[self.config.feature_cols]
                )
                test_df[f"{self.config.pred_col}_fold{n_fold}"] = bst.predict(
                    test_df[self.config.feature_cols]
                )
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance(importance_type="gain"))
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
    ) -> lgb.Booster:
        train_df_ = train_df.query("fold!=@n_fold")
        X_train = train_df_[self.config.feature_cols]
        y_train = train_df_[self.config.label_col]
        X_valid = train_df.query("fold==@n_fold")[self.config.feature_cols]
        y_valid = train_df.query("fold==@n_fold")[self.config.label_col]
        logger.info(
            f"{self.config.label_col} [Fold {n_fold}] train shape: {X_train.shape}, valid shape: {X_valid.shape}"
        )
        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            # weight=train_df.query("fold!=@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            # weight=train_df.query("fold==@n_fold")["weights"].values,
            feature_name=self.config.feature_cols,
        )
        evals_result = {}
        bst = lgb.train(
            dict(self.config.params),
            lgtrain,
            num_boost_round=100000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=self.config.early_stopping_rounds,
            # categorical_feature=self.config.cat_cols,
            verbose_eval=self.config.verbose_eval,
            evals_result=evals_result,
        )
        if self.config.params.objective == "binary":
            best_idx = np.argmax(evals_result["valid"][self.config.params.metric])
        else:
            best_idx = np.argmin(evals_result["valid"][self.config.params.metric])
        logger.info(
            f"best_itelation: {best_idx}, "
            f"train: {evals_result['train'][self.config.params.metric][best_idx]}, "
            f"valid: {evals_result['valid'][self.config.params.metric][best_idx]}"
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class XGBModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, xgb.Booster] = {}

    def custom_metric(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label().astype(float)
        score = average_precision_score(y_true, y_pred)
        return "average_precision_score", -score

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(self.config.n_fold):
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid_df = train_df.query("fold == @n_fold")
            if self.config.params.objective == "multi:softmax":
                preds = bst.predict(xgb.DMatrix(valid_df[self.config.feature_cols]))
                test_preds = bst.predict(xgb.DMatrix(test_df[self.config.feature_cols]))
                for i in range(self.config.params.num_class):
                    train_df.loc[
                        valid_df.index, f"{self.config.label_col}_prob{i}"
                    ] = preds[:, i]
                    test_df[
                        f"{self.config.label_col}_prob{i}_fold{n_fold}"
                    ] = test_preds[:, i]
            else:
                train_df.loc[valid_df.index, self.config.pred_col] = bst.predict(
                    xgb.DMatrix(valid_df[self.config.feature_cols])
                )
                test_df[f"{self.config.pred_col}_fold{n_fold}"] = bst.predict(
                    xgb.DMatrix(test_df[self.config.feature_cols])
                )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_score(importance_type="gain")
            importances.append(
                [
                    importance_dict[col] if col in importance_dict else 0
                    for col in self.config.feature_cols
                ]
            )
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)

        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> xgb.Booster:
        X_train = train_df.query("fold!=@n_fold")[self.config.feature_cols]
        y_train = train_df.query("fold!=@n_fold")[self.config.label_col]

        X_valid = train_df.query("fold==@n_fold")[self.config.feature_cols]
        y_valid = train_df.query("fold==@n_fold")[self.config.label_col]
        print("=" * 10, self.config.label_col, n_fold, "=" * 10)
        dtrain = xgb.DMatrix(
            X_train,
            label=np.array(y_train),
            feature_names=self.config.feature_cols,
        )
        dvalid = xgb.DMatrix(
            X_valid,
            label=np.array(y_valid),
            feature_names=self.config.feature_cols,
        )
        bst = xgb.train(
            self.config.params,
            dtrain,
            num_boost_round=50000,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=100,
            verbose_eval=50,
            # feval=self.custom_metric,
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )


class CatModel(object):
    """
    label_col毎にxgboost modelを作成するようのクラス
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.model_dicts: Dict[int, cat.Booster] = {}

    def custom_metric(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true = dtrain.get_label().astype(float)
        score = average_precision_score(y_true, y_pred)
        return "average_precision_score", -score

    def store_model(self, bst: xgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(self.config.n_fold):
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid_df = train_df.query("fold == @n_fold")
            if self.config.params.loss_function == "multi:softmax":
                preds = bst.predict(cat.Pool(valid_df[self.config.feature_cols]))
                test_preds = bst.predict(cat.Pool(test_df[self.config.feature_cols]))
                for i in range(self.config.params.num_class):
                    train_df.loc[
                        valid_df.index, f"{self.config.label_col}_prob{i}"
                    ] = preds[:, i]
                    test_df[
                        f"{self.config.label_col}_prob{i}_fold{n_fold}"
                    ] = test_preds[:, i]
            else:
                train_df.loc[valid_df.index, self.config.pred_col] = bst.predict(
                    cat.Pool(
                        valid_df[self.config.feature_cols],
                        cat_features=self.config.categorical_features_indices,
                    ),
                )
                test_df[f"{self.config.pred_col}_fold{n_fold}"] = bst.predict(
                    cat.Pool(
                        test_df[self.config.feature_cols],
                        cat_features=self.config.categorical_features_indices,
                    )
                )
            self.store_model(bst, n_fold)
            importance_dict = bst.get_feature_importance()
            importances.append(
                [
                    importance_dict[col] if col in importance_dict else 0
                    for col in self.config.feature_cols
                ]
            )
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std},
            index=self.config.feature_cols,
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)

        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> xgb.Booster:
        X_train = train_df.query("fold!=@n_fold")[self.config.feature_cols]
        y_train = train_df.query("fold!=@n_fold")[self.config.label_col]

        X_valid = train_df.query("fold==@n_fold")[self.config.feature_cols]
        y_valid = train_df.query("fold==@n_fold")[self.config.label_col]
        print("=" * 10, self.config.label_col, n_fold, "=" * 10)
        dtrain = cat.Pool(
            X_train,
            label=np.array(y_train),
            feature_names=self.config.feature_cols,
            cat_features=self.config.categorical_features_indices,
        )
        dvalid = cat.Pool(
            X_valid,
            label=np.array(y_valid),
            feature_names=self.config.feature_cols,
            cat_features=self.config.categorical_features_indices,
        )
        bst = cat.train(
            pool=dtrain,
            params=dict(self.config.params),
            num_boost_round=50000,
            evals=dvalid,
            early_stopping_rounds=100,
            verbose_eval=100,
            # feval=self.custom_metric,
        )
        return bst

    def save_model(self, model_dir: str, suffix: str = "") -> None:
        with open(f"{model_dir}/booster{suffix}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)

    def save_importance(
        self,
        result_path: str,
        suffix: str = "",
    ) -> None:
        self.importance_df.sort_values("mean").iloc[-50:].plot.barh(
            xerr="std", figsize=(10, 20)
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.png",
            )
        )
        self.importance_df.name = "feature_name"
        self.importance_df = self.importance_df.reset_index().sort_values(
            by="mean", ascending=False
        )
        self.importance_df.to_csv(
            os.path.join(
                result_path,
                f"importance_{self.config.label_col + suffix}.csv",
            ),
            index=False,
        )
