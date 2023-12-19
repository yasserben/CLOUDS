"""
This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates.

Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/cityscapes_panoptic.py
"""

import json
import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from . import openseg_classes

CITYSCAPES_CATEGORIES = openseg_classes.get_categories_with_prompt_eng()

"""
This file contains functions to register the Cityscapes panoptic dataset to the DatasetCatalog.
"""


logger = logging.getLogger(__name__)


def load_acdc_semantic(image_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    if "rain" in image_dir:
        with open("datasets/cityscapes_list/acdc_rain_dict.json") as f:
            dataset_dicts = json.load(f)
    elif "snow" in image_dir:
        with open("datasets/cityscapes_list/acdc_snow_dict.json") as f:
            dataset_dicts = json.load(f)
    elif "fog" in image_dir:
        with open("datasets/cityscapes_list/acdc_fog_dict.json") as f:
            dataset_dicts = json.load(f)
    elif "night" in image_dir:
        with open("datasets/cityscapes_list/acdc_night_dict.json") as f:
            dataset_dicts = json.load(f)
    return dataset_dicts


# rename to avoid conflict
_RAW_CITYSCAPES_SEMANTIC_SPLITS = {
    "acdc_night_val": (
        "acdc/rgb_anon/night/val",
        "acdc/gt/night/val",
    ),
    "acdc_rain_val": (
        "acdc/rgb_anon/rain/val",
        "acdc/gt/rain/val",
    ),
    "acdc_fog_val": (
        "acdc/rgb_anon/fog/val",
        "acdc/gt/fog/val",
    ),
    "acdc_snow_val": (
        "acdc/rgb_anon/snow/val",
        "acdc/gt/snow/val",
    ),
}


def register_all_acdc_semantic(root):
    meta = {}

    thing_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SEMANTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        DatasetCatalog.register(
            key,
            lambda x=image_dir: load_acdc_semantic(x),
        )
        MetadataCatalog.get(key).set(
            image_root=image_dir,
            gt_dir=gt_dir,
            evaluator_type="acdc_sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_acdc_semantic(_root)
