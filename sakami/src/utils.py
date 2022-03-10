import functools
import gc
import os
import random
import shutil
import subprocess
import time
from contextlib import ContextDecorator
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from google.cloud import storage
from loguru import logger


def set_seed(seed: int = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class timer(ContextDecorator):
    """Context-manager that logs elapsed time of a process.
    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    Paramters
    ---------
    message : str
        The displayed message.

    Examples
    --------
    - Usage as a context-manager

        >>> with timer("read csv"):
        >>>     train_df = pd.read_csv(TRAIN_PATH)
        [read csv] start.
        [read csv] done in 0.1 min.

    - Usage as a decorator

        >>> @timer()
        >>> def read_csv():
        >>>     train_df = pd.read_csv(TRAIN_PATH)
        >>>     return train_df
        >>>
        >>> train_df = read_csv()
        [read_csv] start.
        [read_csv] done in 0.1 min.
    """

    def __init__(self, message: str = None):
        self.message = message

    def __call__(self, function):
        if self.message is None:
            self.message = function.__name__
        return super().__call__(function)

    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"[{self.message}] start.")

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            elapsed_time = time.time() - self.start_time
            logger.info(f"[{self.message}] done in {elapsed_time / 60:.1f} min.")


def freeze(cls):
    """Decorator function for fixing class variables.

    Examples
    --------

        >>> @freeze
        >>> class config:
        >>>     x = 10
        >>>     y = 20

        >>> config.x = 30
        ValueError: Cannot overwrite config.
    """

    class _Const(type):
        """Metaclass of the configuration class.

        Examples
        --------

            >>> class config(metaclass=_Const):
            >>>     x = 10
            >>>     y = 20

            >>> config.x = 30
            ValueError: Cannot overwrite config.

        References
        ----------
        - https://cream-worker.blog.jp/archives/1077207909.html
        """

        def __setattr__(self, name, value):
            raise ValueError("Cannot overwrite config.")

    class frozen_cls(cls, metaclass=_Const):
        pass

    return frozen_cls


def download_from_gcs(path: str, bucket_name: str, gcs_prefix: str = None):
    """Download a directory from Google Cloud Storage.

    Parameters
    ----------
    path : str
        The local path to the directory to download.
    bucket_name : str
        The GCS bucket name.
    gcs_prefix : str
        The GCS path of the directory you want to download.
    """
    if shutil.which("gsutil") is not None:
        subprocess.run(
            f"gsutil -m cp -r gs://{bucket_name}/{gcs_prefix} {str(path)}",
            shell=True,
        )

    else:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)

        for blob in client.list_blobs(bucket, prefix=gcs_prefix):
            download_path = Path(path) / blob.name
            download_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(filename=str(download_path))


def upload_to_gcs(path: str, bucket_name: str, gcs_prefix: str = ""):
    """Upload a directory to Google Cloud Storage.

    Parameters
    ----------
    path : str
        The local path to the directory you want to upload.
    bucket_name : str
        The GCS bucket name.
    gcs_prefix : str
        The GCS path to the directory to upload.
    """
    if shutil.which("gsutil") is not None:
        subprocess.run(
            f"gsutil -m cp -r {str(path)} gs://{bucket_name}/{gcs_prefix}",
            shell=True,
        )

    else:
        if gcs_prefix != "" and not gcs_prefix.endswith("/"):
            gcs_prefix += "/"

        def _upload_to_gcs(
            bucket: storage.bucket.Bucket, path: Path, dir_name: str = ""
        ):
            if path.is_file():
                file_name = dir_name + path.name
                blob = bucket.blob(file_name)
                blob.upload_from_filename(filename=str(path))
            else:
                for child_path in path.iterdir():
                    _upload_to_gcs(bucket, child_path, dir_name + path.name + "/")

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        _upload_to_gcs(bucket, Path(path), dir_name=gcs_prefix)


def clear_memory(function: Callable):
    """Decorator function for clearing memory cache."""

    @functools.wraps(function)
    def _clear_memory(*args, **kwargs):
        function(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()

    return _clear_memory
