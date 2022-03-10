from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer

from sklearn.preprocessing import LabelEncoder, StandardScaler
from albumentations import (
    GridDistortion,
    ElasticTransform,
    OpticalDistortion,
    Resize,
    RandomCrop,
    HorizontalFlip,
    RGBShift,
    HueSaturationValue,
    GridDropout,
    CoarseDropout,
    ShiftScaleRotate,
    RandomGamma,
    RandomBrightnessContrast,
    Compose,
    OneOf,
    Normalize,
    Downscale,
)


class YogaDataset(object):
    def __init__(self, config: DictConfig, mode: str = "train"):

        self.config = config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load

        train_df = pd.read_csv("./input/train_fold.csv")
        test_df = pd.read_csv("./input/sample_submission.csv")

        self.id_col = config.id_col
        self.label_col = config.label_col
        # preprocessing
        if mode == "train":
            df = train_df.query(f"fold!={config.n_fold}").reset_index(drop=True)
            self.labels = df[self.label_col].map(config.label_map).values
        elif mode == "valid":
            df = train_df.query(f"fold=={config.n_fold}").reset_index(drop=True)
            self.labels = df[self.label_col].map(config.label_map).values
        elif mode == "test":
            df = test_df
        self.ids = df[self.id_col].values

    def set_refinement_step(self):
        self.refinement_step = True

    def _resize_mix(self, image: np.ndarray) -> Tuple[np.ndarray, int, float]:
        alpha, beta = self.config.resize_mix.alpha, self.config.resize_mix.beta
        if (
            np.random.rand() >= 0.5
            and not self.refinement_step
            and self.mode == "train"
        ):
            rand_idx = np.random.randint(len(self))
            rand_image_name, rand_label = self.ids[rand_idx], self.labels[rand_idx]
            rand_image = cv2.imread(
                f"{self.config.train_image_path}/{rand_image_name}"
            )[..., ::-1]
            aug = self._augmenation(rand_image)
            rand_image = aug(image=rand_image)["image"]
            # (alpha, beta)の範囲で比率を算出
            tau = (beta - alpha) * np.random.rand() + alpha
            size = int(tau * rand_image.shape[0])
            rand_image = cv2.resize(rand_image, (size, size))
            # paste
            top_left = np.random.randint(0, image.shape[0] - size)
            image[top_left : top_left + size, top_left : top_left + size] = rand_image
            # ラベルの混ぜ合わせ用
            lam = tau**2
        else:
            rand_label = 0
            lam = 0

        return image, rand_label, lam

    def _augmenation(self, image: np.ndarray, p: float = 0.5) -> Compose:
        aug_list = []
        height, width, _ = image.shape
        if self.mode == "train":
            if not self.refinement_step:
                if height >= 512:
                    height = np.random.randint(height - height * 0.2, height)
                    width = np.random.randint(width - width * 0.2, width)
                else:
                    height = np.random.randint(height - height * 0.1, height)
                    width = np.random.randint(width - width * 0.1, width)

                aug_list.extend(
                    [
                        RandomCrop(height, width, p),
                        HorizontalFlip(p),
                        OneOf(
                            [GridDistortion(), ElasticTransform(), OpticalDistortion()],
                            p,
                        ),
                        OneOf([RGBShift(), HueSaturationValue()], p),
                        OneOf(
                            [
                                RandomBrightnessContrast(
                                    brightness_limit=0.1, contrast_limit=0.8
                                ),
                                RandomGamma(),
                            ],
                            p,
                        ),
                        # OneOf([GridDropout(), CoarseDropout()], p),
                        ShiftScaleRotate(
                            shift_limit=0.2, scale_limit=0.2, rotate_limit=90, p=p
                        ),
                        Downscale(scale_min=0.8, scale_max=0.99, p=p),
                    ]
                )

        aug_list.append(
            Resize(
                self.config.image_size,
                self.config.image_size,
                cv2.INTER_LINEAR,
            ),
        )
        if self.config.normalize == "swin":
            aug_list.append(
                Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                    max_pixel_value=255,
                    p=1,
                )
            )
        elif self.config.normalize == "imagenet":
            aug_list.append(
                Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                    max_pixel_value=255,
                    p=1,
                )
            )
        elif self.config.normalize == "image_wise":
            mean_ = [image[:, :, i].mean() for i in range(3)]
            std_ = [image[:, :, i].std() for i in range(3)]
            aug_list.append(
                Normalize(
                    mean_,
                    std_,
                    max_pixel_value=1,
                    p=1,
                )
            )
        else:
            raise NotImplementedError(
                f"{self.config.normalize} mode is not Implemented."
            )
        return Compose(aug_list)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # feature = np.vstack(self.features[idx]).T.astype(np.float32)

        if self.mode != "test":
            img = cv2.imread(f"./input/images/train_images/{self.ids[idx]}")[..., ::-1]
            aug = self._augmenation(img)
            img = aug(image=img)["image"]
            label = self.labels[idx]
            return {
                self.id_col: self.ids[idx],
                "image": img,
                self.label_col: label,
            }
        else:
            img = cv2.imread(f"./input/images/test_images/{self.ids[idx]}")[..., ::-1]
            aug = self._augmenation(img)
            img = aug(image=img)["image"]
            return {
                self.id_col: self.ids[idx],
                "image": img,
            }


class DelhiDataset(object):
    def __init__(self, config: DictConfig, mode: str = "train"):

        self.config = config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load

        train_df = pd.read_csv(config.train_path)
        test_df = pd.read_csv(config.test_path)

        self.id_col = config.id_col
        self.label_col = config.label_col
        # preprocessing
        self.numerical_cols = [
            col
            for col in config.feature.feature_cols
            if col not in ["id"] + list(config.feature.cat_cols)
        ]
        self.cat_cols = config.feature.cat_cols
        for col1, col2 in combinations(["unit", "categoryA", "categoryE"], 2):
            train_df[f"{col1}_{col2}"] = train_df[[col1, col2]].apply(
                lambda xs: "_".join([str(x) for x in xs.values]), axis=1
            )
            test_df[f"{col1}_{col2}"] = test_df[[col1, col2]].apply(
                lambda xs: "_".join([str(x) for x in xs.values]), axis=1
            )
            if config.process_unseen:
                overlap_cat = set(train_df[f"{col1}_{col2}"].unique()) & set(
                    test_df[f"{col1}_{col2}"].unique()
                )

                # 被ってるカテゴリ以外その他とする
                train_df.loc[
                    train_df.query(f"{col1}_{col2} not in @overlap_cat").index,
                    f"{col1}_{col2}",
                ] = "other"
                test_df.loc[
                    test_df.query(f"{col1}_{col2} not in @overlap_cat").index,
                    f"{col1}_{col2}",
                ] = "other"
            self.cat_cols += [f"{col1}_{col2}"]

        for col in self.numerical_cols:
            std = StandardScaler()
            mean_ = np.nanmean(
                np.concatenate([train_df[col].values, test_df[col].values])
            )
            train_df[col].fillna(mean_, inplace=True)
            test_df[col].fillna(mean_, inplace=True)
            train_df[col] = std.fit_transform(train_df[[col]])
            test_df[col] = std.transform(test_df[[col]])

        for col in self.cat_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                train_df[col].fillna(-20000, inplace=True)
                test_df[col].fillna(-20000, inplace=True)
            else:
                train_df[col].fillna("None", inplace=True)
                test_df[col].fillna("None", inplace=True)
            all_cat = pd.concat([train_df[col], test_df[col]], axis=0)
            le = LabelEncoder()
            le.fit(all_cat)
            train_df[col] = le.transform(train_df[[col]])
            test_df[col] = le.transform(test_df[[col]])

        if mode == "train":
            df = train_df.query(f"fold!={config.n_fold}").reset_index(drop=True)
            self.labels = np.log1p(df[self.label_col].values)
        elif mode == "valid":
            df = train_df.query(f"fold=={config.n_fold}").reset_index(drop=True)
            self.labels = np.log1p(df[self.label_col].values)
        elif mode == "test":
            df = test_df
        self.ids = df[self.id_col].values
        self.df = df

    def set_refinement_step(self):
        self.refinement_step = True

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature = self.df.loc[idx, self.numerical_cols].values.astype(np.float32)
        cat_feature_dict = self.df.loc[idx, self.cat_cols].to_dict()
        out = {
            self.id_col: self.ids[idx],
            "feature": feature,
        }
        out.update(cat_feature_dict)
        if self.mode != "test":
            out.update({self.label_col: self.labels[idx]})
            return out
        return out


class TextDataset(object):
    def __init__(self, config: DictConfig, mode: str = "train"):

        self.config = config
        self.mode = mode
        self.refinement_step = False

        # Data load
        train_df = pd.read_csv("./input/train_fold.csv")
        test_df = pd.read_csv("./input/test_clean.csv")
        for col in config.feature.feature_cols:
            std = StandardScaler()
            mean_ = np.nanmean(
                np.concatenate([train_df[col].values, test_df[col].values])
            )
            train_df[col].fillna(mean_, inplace=True)
            test_df[col].fillna(mean_, inplace=True)
            train_df[col] = std.fit_transform(train_df[[col]])
            test_df[col] = std.transform(test_df[[col]])

        for col in config.feature.cat_cols:
            if pd.api.types.is_numeric_dtype(train_df[col]):
                train_df[col].fillna(-20000, inplace=True)
                test_df[col].fillna(-20000, inplace=True)
            else:
                train_df[col].fillna("None", inplace=True)
                test_df[col].fillna("None", inplace=True)
            all_cat = pd.concat([train_df[col], test_df[col]], axis=0)
            le = LabelEncoder()
            le.fit(all_cat)
            train_df[col] = le.transform(train_df[[col]])
            test_df[col] = le.transform(test_df[[col]])

        self.id_col = config.id_col
        self.label_col = config.label_col
        # preprocessing
        if mode == "train":
            df = train_df.query(f"fold!={config.n_fold}").reset_index(drop=True)
            self.labels = df[self.label_col].map(config.label_map).values
        elif mode == "valid":
            df = train_df.query(f"fold=={config.n_fold}").reset_index(drop=True)
            self.labels = df[self.label_col].map(config.label_map).values
        elif mode == "test":
            df = test_df
        self.ids = df[self.id_col].values
        self.texts1 = df["s1"].to_list()
        self.texts2 = df["s2"].to_list()
        self.features = df[config.feature.feature_cols].values
        self.cat_features = df[config.feature.cat_cols].values
        self.tokenizer = AutoTokenizer.from_pretrained(config.text.backbone)

    def set_refinement_step(self):
        self.refinement_step = True

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # feature = np.vstack(self.features[idx]).T.astype(np.float32)
        max_length = 256
        text1, text2 = self.texts1[idx], self.texts2[idx]
        encode = self.tokenizer(text1, text2)
        input_ids = encode["input_ids"]
        attention_mask = encode["attention_mask"]
        raw_length = len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (
            max_length - raw_length
        )
        attention_mask = attention_mask + [0] * (max_length - raw_length)
        if self.mode != "test":
            label = self.labels[idx]
            return {
                self.id_col: self.ids[idx],
                "input_ids": np.array(input_ids),
                "attention_mask": np.array(attention_mask),
                "feature": self.features[idx],
                "lang_code": self.cat_features[idx],
                "raw_length": raw_length,
                self.label_col: label,
            }
        else:
            return {
                self.id_col: self.ids[idx],
                "input_ids": np.array(input_ids),
                "attention_mask": np.array(attention_mask),
                "feature": self.features[idx],
                "lang_code": self.cat_features[idx],
                "raw_length": raw_length,
            }


def get_dataset(config, mode):
    print("dataset class:", config.dataset_class)
    f = globals().get(config.dataset_class)
    return f(config, mode)
