import torch
from PIL import Image
import os
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
import json
from . import openseg_classes
CITYSCAPES_CATEGORIES = openseg_classes.get_categories_with_prompt_eng()


def my_gta5_dataset_function():
    """
    This function is called by detectron2.data.DatasetCatalog.register()
    """

    with open("datasets/gta_list/train_dict.json") as f:
        dataset_dicts = json.load(f)
    return dataset_dicts


DatasetCatalog.register("gta5", my_gta5_dataset_function)

MetadataCatalog.get("gta5").set(
    stuff_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
)
MetadataCatalog.get("gta5").set(
    stuff_colors=[
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
)

MetadataCatalog.get("gta5").set(
    thing_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
)
MetadataCatalog.get("gta5").set(
    thing_colors=[
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
)

MetadataCatalog.get("gta5").set(ignore_label=255)
