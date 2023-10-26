"""
Utility methods for datasets.
"""
from __future__ import annotations

import hashlib
import os
import sys
from os.path import isfile
from typing import Any, List, Set, Tuple, Union

import numpy as np
import requests
from numpy.typing import ArrayLike


def download_progress_bar(file_url: str, file_dest: str):
    """
    Download while outputting a progress bar.
    Modified from https://sumit-ghosh.com/articles/python-download-progress-bar/

    Args:
        file_url (str): url to download from
        file_dest (str): path at which to save downloaded file
    """

    with open(file_dest, "wb") as f:
        response = requests.get(file_url, stream=True)
        total = response.headers.get("content-length")

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)

            print("Downloading dataset")
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write(
                    "\r[{}{}] {:.0f}%".format(
                        "█" * done, "." * (50 - done), float(downloaded / total) * 100
                    )
                )
                sys.stdout.flush()

    sys.stdout.write("\n")


# from TorchVision
# https://github.com/pytorch/vision/blob/48f8473e21b0f3e425aabc60db201b68fedf59b3/torchvision/datasets/utils.py#L51-L66  # noqa: E501
def _calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but
    # indicates that we are not using the MD5 checksum for cryptography. This enables its usage
    # in restricted environments like FIPS.
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        # switch to simpler assignment operator once we support only Python >=3.8
        # while chunk := f.read(chunk_size):
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    fmd5 = _calculate_md5(fpath, **kwargs)
    return (md5 == fmd5), fmd5


def _getZenodoFileURL(record_id: int, file_name: str) -> str:
    """Finds URL and md5 hash for downloading the file ``file_name`` from a Zenodo record."""

    import requests

    records_url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(records_url).json()

    # Zenodo API seems to be switching back and forth between these at the moment... so trying both
    try:
        file = next(item for item in r["files"] if item["filename"] == file_name)
        file_url = file["links"]["download"]
        md5 = file["checksum"]
    except KeyError:
        file = next(item for item in r["files"] if item["key"] == file_name)
        file_url = file["links"]["self"]
        md5 = file["checksum"].split("md5:")[1]

    return file_url, md5


def checkDownloadZenodoDataset(
    data_dir: str, dataset_name: str, record_id: int, key: str, download: bool
) -> str:
    """
    Checks if dataset exists and md5 hash matches;
    if not and download = True, downloads it from Zenodo, and returns the file path.
    or if not and download = False, raises an error.
    """
    file_path = f"{data_dir}/{key}"
    file_url, md5 = _getZenodoFileURL(record_id, key)

    if download:
        if isfile(file_path):
            match_md5, fmd5 = _check_md5(file_path, md5)
            if not match_md5:
                print(
                    f"File corrupted - MD5 hash of {file_path} does not match: "
                    f"(expected md5:{md5}, got md5:{fmd5}), "
                    "removing existing file and re-downloading."
                    "\nPlease open an issue at https://github.com/jet-net/JetNet/issues/new "
                    "if you believe this is an error."
                )
                os.remove(file_path)

        if not isfile(file_path):
            os.makedirs(data_dir, exist_ok=True)

            print(f"Downloading {dataset_name} dataset to {file_path}")
            download_progress_bar(file_url, file_path)

    if not isfile(file_path):
        raise RuntimeError(
            f"Dataset {dataset_name} not found at {file_path}, "
            "you can use download=True to download it."
        )

    match_md5, fmd5 = _check_md5(file_path, md5)
    if not match_md5:
        raise RuntimeError(
            f"File corrupted - MD5 hash of {file_path} does not match: "
            f"(expected md5:{md5}, got md5:{fmd5}), "
            "you can use download=True to re-download it."
            "\nPlease open an issue at https://github.com/jet-net/JetNet/issues/new "
            "if you believe this is an error."
        )

    return file_path


def getOrderedFeatures(
    data: ArrayLike, features: List[str], features_order: List[str]
) -> np.ndarray:
    """Returns data with features in the order specified by ``features``.

    Args:
        data (ArrayLike): input data
        features (List[str]): desired features in order
        features_order (List[str]): name and ordering of features in input data

    Returns:
        (np.ndarray): data with features in specified order
    """

    if np.all(features == features_order):  # check if already in order
        return data

    ret_data = []
    for feat in features:
        assert (
            feat in features_order
        ), f"`{feat}` feature does not exist in this dataset (available features: {features_order})"
        index = features_order.index(feat)
        ret_data.append(data[..., index, np.newaxis])

    return np.concatenate(ret_data, axis=-1)


def checkStrToList(
    *inputs: List[Union[str, List[str], Set[str]]], to_set: bool = False
) -> Union[List[List[str]], List[Set[str]], list]:
    """Converts str inputs to a list or set"""
    ret = []
    for inp in inputs:
        if isinstance(inp, str):
            inp = [inp] if not to_set else {inp}
        ret.append(inp)

    return ret if len(inputs) > 1 else ret[0]


def checkListNotEmpty(*inputs: List[list]) -> List[bool]:
    """Checks that list inputs are not None or empty"""
    ret = []
    for inp in inputs:
        ret.append(inp is not None and len(inp))

    return ret if len(inputs) > 1 else ret[0]


def firstNotNoneElement(*inputs: List[Any]) -> Any:
    """Returns the first element out of all inputs which isn't None"""
    for inp in inputs:
        if inp is not None:
            return inp


def checkConvertElements(
    elem: Union[str, List[str]], valid_types: List[str], ntype: str = "element"
):
    """Checks if elem(s) are valid and if needed converts into a list"""
    if elem != "all":
        elem = checkStrToList(elem, to_set=True)

        for j in elem:
            assert j in valid_types, f"{j} is not a valid {ntype}, must be one of {valid_types}"

    else:
        elem = valid_types

    return elem


def getSplitting(
    length: int, split: str, splits: List[str], split_fraction: List[float]
) -> Tuple[int, int]:
    """
    Returns starting and ending index for splitting a dataset of length ``length`` according to
    the input ``split`` out of the total possible ``splits`` and a given ``split_fraction``.

    "all" is considered a special keyword to mean the entire dataset - it cannot be used to define a
    normal splitting, and if it is a possible splitting it must be the last entry in ``splits``.

    e.g. for ``length = 100``, ``split = "valid"``, ``splits = ["train", "valid", "test"]``,
    ``split_fraction = [0.7, 0.15, 0.15]``

    This will return ``(70, 85)``.
    """

    assert split in splits, f"{split} not a valid splitting, must be one of {splits}"

    if "all" in splits:
        if split == "all":
            return 0, length
        else:
            assert splits[-1] == "all", "'all' must be last entry in ``splits`` array"
            splits = splits[:-1]

    assert np.sum(split_fraction) <= 1.0, "sum of split fractions must be ≤ 1"

    split_index = splits.index(split)
    cuts = (np.cumsum(np.insert(split_fraction, 0, 0)) * length).astype(int)
    return cuts[split_index], cuts[split_index + 1]
