import json
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from . import openseg_classes
CITYSCAPES_CATEGORIES = openseg_classes.get_categories_with_prompt_eng()


def my_bdd_dataset_function():
    """
    This function is called by detectron2.data.DatasetCatalog.register()
    """
    with open("datasets/bdd_list/train_dict.json") as f:
        dataset_dicts = json.load(f)
    return dataset_dicts


DatasetCatalog.register("bdd_val", my_bdd_dataset_function)

MetadataCatalog.get("bdd_val").set(
    stuff_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
)
MetadataCatalog.get("bdd_val").set(
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

MetadataCatalog.get("bdd_val").set(
    thing_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
)
MetadataCatalog.get("bdd_val").set(
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


MetadataCatalog.get("bdd_val").set(ignore_label=255)
MetadataCatalog.get("bdd_val").set(image_dir="datasets/bdd100k/images/10k/val")
MetadataCatalog.get("bdd_val").set(gt_dir="datasets/bdd100k/labels/sem_seg/masks/val")
MetadataCatalog.get("bdd_val").set(evaluator_type="bdd_sem_seg")
