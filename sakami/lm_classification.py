import ast
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import transformers
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoTokenizer,
    BatchEncoding,
    BertModel,
    BertPreTrainedModel,
    PreTrainedTokenizerBase,
    get_scheduler,
)
from transformers.file_utils import PaddingStrategy

from src.data import BucketSampler
from src.utils import freeze, set_seed, timer, upload_to_gcs


@freeze
class config:
    id_column = "id"
    text_column_1 = "s1"
    text_column_2 = "s2"
    target_column = "category"
    prediction_column = "category"

    model_name = "bert-base-multilingual-uncased"
    use_fast = True
    fp16 = True

    n_splits = 4
    n_epochs = 3

    max_length = 220
    batch_size = 32
    eval_batch_size = 64
    bucket_size = 100

    warmup = 0.05
    scheduler = "linear"
    lr = 3e-5

    bucket_name = "kaggledays_championship"
    bucket_path = "mumbai/sakami/bert_v10/"  # CHECK HERE!!!

    device = torch.device("cuda")
    seed = 1029


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path("./input/we-are-all-alike-on-the-inside/")
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_df, test_df


def preprocess_text(texts: List[str]) -> List[str]:
    res = []
    for text in tqdm(texts, desc="preprocess"):
        if text[0] == "[":
            if "'" in text:
                text = text.replace("'", "")
            if text[1] != "'":
                text = "['" + text[1:-1] + "']"
            res.append(" ".join(ast.literal_eval(text)))
        else:
            res.append(text)
    return res


def generate_split(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    train_fold = pd.read_csv("./input/train_fold.csv")

    if train_fold["fold"].nunique() != config.n_splits:
        raise ValueError("The number of splits in CSV and config must be the same.")

    if len(train_fold) != len(X):
        train_fold = train_fold.sample(len(X), random_state=config.seed)

    for i in range(config.n_splits):
        train_idx = np.where(train_fold["fold"] != i)[0]
        valid_idx = np.where(train_fold["fold"] == i)[0]
        yield train_idx, valid_idx


class TextDataset(Dataset):
    def __init__(
        self,
        encodings: List[BatchEncoding],
        target: Optional[np.ndarray] = None,
        indices: Optional[List[int]] = None,
    ):
        if indices is None:
            indices = np.arange(len(encodings))

        self.encodings = encodings
        self.target = target
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[BatchEncoding, float]:
        data = (
            idx,
            self.encodings[self.indices[idx]],
        )
        if self.target is not None:
            data += (self.target[self.indices[idx]],)
        return data


@dataclass
class PaddingCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    device: Union[torch.device, str] = "cpu"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batches = list(zip(*features))
        x_batch = self.tokenizer.pad(
            batches[1],
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in x_batch:
            x_batch["labels"] = x_batch["label"]
            del x_batch["label"]
        if "label_ids" in x_batch:
            x_batch["labels"] = x_batch["label_ids"]
            del x_batch["label_ids"]

        x_batch = {k: v.to(self.device) for k, v in x_batch.items()}

        if len(batches) == 3:
            y_batch = torch.tensor(batches[2]).to(self.device)
            return x_batch, y_batch

        indices = list(batches[0])
        return indices, x_batch


class BertClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 4, 3)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        pooled_output = torch.cat(
            [
                outputs.hidden_states[-4][:, 0],
                outputs.hidden_states[-3][:, 0],
                outputs.hidden_states[-2][:, 0],
                outputs.hidden_states[-1][:, 0],
            ],
            dim=-1,
        )
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        self.eval()
        dtype = torch.half if config.fp16 else torch.float
        preds = torch.zeros((len(data_loader.dataset), 3), dtype=dtype, device=self.device)
        with torch.no_grad():
            for idx, batch in tqdm(data_loader, desc="predict", leave=False):
                with autocast(enabled=config.fp16):
                    preds[idx] = self(**batch).detach()
        return preds.float().softmax(axis=1).cpu().numpy()


def main(debug: bool = False):
    start_time = time.time()
    set_seed(config.seed)
    transformers.logging.set_verbosity_error()

    # load data
    train_df, test_df = load_data()

    if debug:
        train_df = train_df.sample(100, random_state=config.seed).reset_index(drop=True)
        test_df = test_df.sample(100, random_state=config.seed).reset_index(drop=True)

    # preprocess
    train_df["s1"] = preprocess_text(train_df["s1"].to_list())
    test_df["s1"] = preprocess_text(test_df["s1"].to_list())

    logger.info(f"train shape : {train_df.shape}")
    logger.info(f"test shape  : {test_df.shape}")

    le = LabelEncoder()
    train_df[config.target_column] = le.fit_transform(train_df[config.target_column])

    # prepare data
    with timer("prepare data"):
        # encode text
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=config.use_fast)
        train_encodings = []
        train_sort_keys = []
        for text1, text2 in zip(
            tqdm(train_df[config.text_column_1], desc="train encode text"), train_df[config.text_column_2]
        ):
            encoding = tokenizer.encode_plus(text1, text2, max_length=config.max_length, truncation="longest_first")
            train_encodings.append(encoding)
            train_sort_keys.append(len(encoding.input_ids))

        # collate function
        collator = PaddingCollator(tokenizer, device=config.device)

        # test
        test_encodings = []
        test_sort_keys = []
        for text1, text2 in zip(
            tqdm(test_df[config.text_column_1], desc="test encode text"), test_df[config.text_column_2]
        ):
            encoding = tokenizer.encode_plus(text1, text2, max_length=config.max_length, truncation="longest_first")
            test_encodings.append(encoding)
            test_sort_keys.append(len(encoding.input_ids))

        test_dataset = TextDataset(test_encodings)
        test_sampler = BucketSampler(
            test_dataset,
            test_sort_keys,
            bucket_size=None,
            batch_size=config.eval_batch_size,
            shuffle_data=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.eval_batch_size,
            sampler=test_sampler,
            num_workers=0,
            collate_fn=collator,
        )

    # token length summary
    train_sort_keys = np.array(train_sort_keys)
    test_sort_keys = np.array(test_sort_keys)

    logger.info("token length statistics")
    logger.info(f"  max   : {train_sort_keys.max()}")
    logger.info(f"  mean  : {train_sort_keys.mean()}")
    logger.info(f"  50%   : {np.percentile(train_sort_keys, 50)}")
    logger.info(f"  75%   : {np.percentile(train_sort_keys, 75)}")
    logger.info(f"  90%   : {np.percentile(train_sort_keys, 90)}")
    logger.info(f"  95%   : {np.percentile(train_sort_keys, 95)}")
    logger.info(f"  99%   : {np.percentile(train_sort_keys, 99)}")
    logger.info(f"  99.5% : {np.percentile(train_sort_keys, 99.5)}")
    logger.info(f"  99.9% : {np.percentile(train_sort_keys, 99.9)}")

    # train
    with timer("train"):
        valid_preds = np.zeros((len(train_df), 3), dtype=np.float64)
        test_preds = np.zeros((len(test_df), 3), dtype=np.float64)
        cv_scores = []

        for fold, (train_idx, valid_idx) in enumerate(generate_split(train_encodings)):
            logger.info("-" * 40)
            logger.info(f"fold {fold + 1}")

            # model
            model = BertClassifier.from_pretrained(config.model_name)
            model.zero_grad()
            model.to(config.device)

            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": 0.01,
                },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=config.lr)

            num_training_steps = len(train_idx) * config.n_epochs // config.batch_size
            num_warmup_steps = int(config.warmup * num_training_steps)
            scheduler = get_scheduler(config.scheduler, optimizer, num_warmup_steps, num_training_steps)
            scaler = GradScaler()

            # data
            train_dataset = TextDataset(train_encodings, train_df[config.target_column], indices=train_idx)
            valid_dataset = TextDataset(train_encodings, indices=valid_idx)

            train_sampler = BucketSampler(
                train_dataset,
                train_sort_keys[train_idx],
                bucket_size=config.bucket_size,
                batch_size=config.batch_size,
                shuffle_data=True,
            )
            valid_sampler = BucketSampler(
                valid_dataset,
                train_sort_keys[valid_idx],
                bucket_size=None,
                batch_size=config.eval_batch_size,
                shuffle_data=False,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=train_sampler,
                num_workers=0,
                collate_fn=collator,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=config.eval_batch_size,
                sampler=valid_sampler,
                num_workers=0,
                collate_fn=collator,
            )

            valid_y = train_df[config.target_column].values[valid_idx]
            loss_ema = None

            for epoch in range(config.n_epochs):
                epoch_start_time = time.time()
                model.train()

                progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
                for x_batch, y_batch in progress:
                    with autocast(enabled=config.fp16):
                        y_preds = model(**x_batch)
                        loss = nn.CrossEntropyLoss()(y_preds, y_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                    loss_ema = loss_ema * 0.9 + loss.item() * 0.1 if loss_ema is not None else loss.item()
                    progress.set_postfix(loss=loss_ema)

                valid_fold_preds = model.predict(valid_loader)
                valid_score = f1_score(valid_y, valid_fold_preds.argmax(axis=1), average="micro")
                epoch_elapsed_time = (time.time() - epoch_start_time) / 60
                logger.info(
                    f"  Epoch {epoch + 1}"
                    f" \t train loss: {loss_ema:.5f}"
                    f" \t valid score: {valid_score:.5f}"
                    f" \t time: {epoch_elapsed_time:.2f} min"
                )

                if debug:
                    break

            valid_preds[valid_idx] = valid_fold_preds
            test_preds += model.predict(test_loader) / config.n_splits
            cv_scores.append(valid_score)

            break

    # save
    save_dir = Path("./output_v10/")
    save_dir.mkdir(exist_ok=True)

    valid_out_df = pd.DataFrame()
    valid_out_df[config.id_column] = train_df[config.id_column]
    valid_out_df[le.classes_] = valid_preds
    valid_out_df.to_csv(save_dir / "valid.csv", index=False)

    test_out_df = pd.DataFrame()
    test_out_df[config.id_column] = test_df[config.id_column]
    test_out_df[le.classes_] = test_preds
    test_out_df.to_csv(save_dir / "test.csv", index=False)

    # upload to GCS
    if not debug:
        upload_to_gcs(save_dir, bucket_name=config.bucket_name, gcs_prefix=config.bucket_path)

    cv_score = np.mean(cv_scores)
    logger.info(f"cv score: {cv_score:.5f}")
    elapsed_time = (time.time() - start_time) / 60
    logger.info(f"all processes done in {elapsed_time:.1f} min.")


if __name__ == "__main__":
    # debug
    logger.info("********************** mode : debug **********************")
    main(debug=True)

    logger.info("-" * 40)
    logger.info("********************** mode : main **********************")
    main(debug=False)
