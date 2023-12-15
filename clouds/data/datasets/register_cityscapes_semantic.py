"""
Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
Licensed under the Apache License, Version 2.0

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


def load_cityscapes_semantic(image_dir):
    if "train" in image_dir:
        with open("datasets/cityscapes_list/train_dict.json") as f:
            dataset_dicts = json.load(f)
    elif "val" in image_dir:
        with open("datasets/cityscapes_list/val_dict.json") as f:
            dataset_dicts = json.load(f)
    return dataset_dicts


# rename to avoid conflict
_RAW_CITYSCAPES_SEMANTIC_SPLITS = {
    "cityscapes_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/train",
    ),
    "cityscapes_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
    ),
}


def register_all_cityscapes_semantic(root):
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
            lambda x=image_dir: load_cityscapes_semantic(
                x,
            ),
        )

        MetadataCatalog.get(key).set(
            image_root=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_semantic(_root)
