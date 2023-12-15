import torch
from PIL import Image
import os
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from . import openseg_classes
import json
CITYSCAPES_CATEGORIES = openseg_classes.get_categories_with_prompt_eng()


def my_synthia_dataset_function():
    """
    This function is called by detectron2.data.DatasetCatalog.register()
    """
    # register my custom synthia dataset to have builtin support in detectron2
    with open(os.path.join("datasets/synthia_list", "train_dict.json")) as f:
        dataset_dicts = json.load(f)
    return dataset_dicts



DatasetCatalog.register("synthia", my_synthia_dataset_function)
MetadataCatalog.get("synthia").set(
    stuff_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
)
MetadataCatalog.get("synthia").set(
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

MetadataCatalog.get("synthia").set(
    thing_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
)
MetadataCatalog.get("synthia").set(
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

MetadataCatalog.get("synthia").set(ignore_label=255)
