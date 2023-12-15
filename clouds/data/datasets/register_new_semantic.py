"""
Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
Licensed under the Apache License, Version 2.0

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/datasets/register_cityscapes_vistas_panoptic.py
"""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from . import openseg_classes

CITYSCAPES_SEM_SEG_CATEGORIES = (
    openseg_classes.get_categories_with_prompt_eng()
)

def load_cityscapes_json():
    with open("datasets/cityscapes_list/val_dict.json") as f:
        dataset_dicts = json.load(f)
    return dataset_dicts


def register_cityscapes(
    name,
    metadata,
    image_root,
    semantic_root,
):
    DatasetCatalog.register(
        name,
        lambda: load_cityscapes_json(
        ),
    )
    MetadataCatalog.get(name).set(
        image_root=image_root,
        gt_dir=semantic_root,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,  # different from other datasets, cityscapes Vistas sets ignore_label to 65
        **metadata,
    )

_RAW_CITYSCAPES_SEMANTIC_SPLITS = {
    "cityscapes_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/gtFine/train",
    ),
    "cityscapes_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/val",
    ),
}


def get_metadata():
    meta = {}
    thing_classes = [k["name"] for k in CITYSCAPES_SEM_SEG_CATEGORIES]
    thing_colors = [k["color"] for k in CITYSCAPES_SEM_SEG_CATEGORIES]
    stuff_classes = [k["name"] for k in CITYSCAPES_SEM_SEG_CATEGORIES]
    stuff_colors = [k["color"] for k in CITYSCAPES_SEM_SEG_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    return meta


def register_all_cityscapes(root):
    metadata = get_metadata()
    for (
        prefix,
        (image_root, semantic_root),
    ) in _RAW_CITYSCAPES_SEMANTIC_SPLITS.items():
        register_cityscapes(
            prefix,
            metadata,
            os.path.join(root, image_root),
            os.path.join(root, semantic_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes(_root)
