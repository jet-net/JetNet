"""
Utility methods for datasets.
"""
from __future__ import annotations
from typing import Set, List, Tuple, Union, Any
from numpy.typing import ArrayLike

import requests
import sys
import os
from os.path import exists

import numpy as np

import logging


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


def checkDownloadZenodoDataset(data_dir: str, dataset_name: str, record_id: int, key: str):
    """Checks if dataset exists, if not downloads it from Zenodo, and returns the file path"""
    file_path = f"{data_dir}/{key}"
    if not exists(file_path):
        os.system(f"mkdir -p {data_dir}")
        file_url = getZenodoFileURL(record_id, key)

        print(f"Downloading {dataset_name} dataset to {file_path}")
        download_progress_bar(file_url, file_path)

    return file_path


def getZenodoFileURL(record_id: int, file_name: str) -> str:
    """Finds URL for downloading the file ``file_name`` from a Zenodo record."""

    import requests

    records_url = f"https://zenodo.org/api/records/{record_id}"
    r = requests.get(records_url).json()
    file_url = next(item for item in r["files"] if item["key"] == file_name)["links"]["self"]
    return file_url


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
            assert splits[-1] == "all", f"'all' must be last entry in ``splits`` array"
            splits = splits[:-1]

    assert np.sum(split_fraction) <= 1.0, "sum of split fractions must be ≤ 1"

    split_index = splits.index(split)
    cuts = (np.cumsum(np.insert(split_fraction, 0, 0)) * length).astype(int)
    return cuts[split_index], cuts[split_index + 1]
