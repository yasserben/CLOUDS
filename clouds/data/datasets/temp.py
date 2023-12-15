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


# def get_cityscapes_semantic_files(image_dir, gt_dir, json_info):
#     files = []
#     # scan through the directory
#     cities = PathManager.ls(image_dir)
#     logger.info(f"{len(cities)} cities found in '{image_dir}'.")
#     image_dict = {}
#     for city in cities:
#         city_img_dir = os.path.join(image_dir, city)
#         for basename in PathManager.ls(city_img_dir):
#             image_file = os.path.join(city_img_dir, basename)
#
#             suffix = "_leftImg8bit.png"
#             assert basename.endswith(suffix), basename
#             basename = os.path.basename(basename)[: -len(suffix)]
#
#             image_dict[basename] = image_file
#
#     for ann, im in zip(json_info["annotations"], json_info["images"]):
#         image_file = image_dict.get(ann["image_id"], None)
#         assert image_file is not None, "No image {} found for annotation {}".format(
#             ann["image_id"], ann["file_name"]
#         )
#         label_file = os.path.join(gt_dir, ann["file_name"])
#         segments_info = ann["segments_info"]
#         height = im["height"]
#         width = im["width"]
#         files.append((image_file, label_file, segments_info, height, width))
#
#     assert len(files), "No images found in {}".format(image_dir)
#     assert PathManager.isfile(files[0][0]), files[0][0]
#     assert PathManager.isfile(files[0][1]), files[0][1]
#     return files
#
#
# def load_cityscapes_semantic(image_dir, gt_dir, gt_json, meta):
#     """
#     Args:
#         image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
#         gt_dir (str): path to the raw annotations. e.g.,
#             "~/cityscapes/gtFine/cityscapes_panoptic_train".
#         gt_json (str): path to the json file. e.g.,
#             "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
#         meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
#             and "stuff_dataset_id_to_contiguous_id" to map category ids to
#             contiguous ids for training.
#
#     Returns:
#         list[dict]: a list of dicts in Detectron2 standard format. (See
#         `Using Custom Datasets </tutorials/datasets.html>`_ )
#     """
#
#     assert os.path.exists(
#         gt_json
#     ), "Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files."  # noqa
#     with open(gt_json) as f:
#         json_info = json.load(f)
#     files = get_cityscapes_semantic_files(image_dir, gt_dir, json_info)
#     ret = []
#     for image_file, label_file, segments_info, height, width in files:
#         sem_label_file = (
#             image_file.replace("leftImg8bit", "gtFine").split(".")[0]
#             + "_labelTrainIds.png"
#         )
#         ret.append(
#             {
#                 "file_name": image_file,
#                 "height": height,
#                 "width": width,
#                 "image_id": "_".join(
#                     os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
#                 ),
#                 "sem_seg_file_name": sem_label_file,
#             }
#         )
#     assert len(ret), f"No images found in {image_dir}!"
#     assert PathManager.isfile(
#         ret[0]["sem_seg_file_name"]
#     ), "Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py"  # noqa
#
#     # Writing the dictionary to a JSON file
#     # with open("/home/ids/benigmim/projects/clouds/datasets/cityscapes_list/train_dict.json", 'w') as json_file:
#     #     json.dump(ret, json_file)
#     return ret

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
        # "cityscapes/gtFine/cityscapes_panoptic_train.json",
    ),
    "cityscapes_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
        # "cityscapes/gtFine/cityscapes_panoptic_val.json",
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
            lambda x=image_dir : load_cityscapes_semantic(
                x,
            ),
        )

        MetadataCatalog.get(key).set(
            image_root=image_dir,
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_cityscapes_semantic(_root)
