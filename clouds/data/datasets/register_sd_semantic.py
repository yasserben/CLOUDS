import json
import os
from PIL import Image
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog


from . import openseg_classes

CITYSCAPES_CATEGORIES = openseg_classes.get_categories_with_prompt_eng()


def my_sd_dataset_function(folder_name):
    """
    This function is called by detectron2.data.DatasetCatalog.register()
    """
    # register my custom sd_v1 dataset to have builtin support in detectron2
    root_path = os.path.join("datasets/stable_diffusion", folder_name)
    # check if a file train_dict.json is present or not
    if os.path.isfile(os.path.join(root_path, "train_dict.json")):
        with open(os.path.join(root_path, "train_dict.json")) as f:
            dataset_dicts = json.load(f)
        return dataset_dicts
    else:
        # iterate over all files inside the folder root
        init_list = []
        for root, dirs, files in os.walk(os.path.join(root_path, "images")):
            for file in files:
                if file.endswith(".png"):
                    if os.path.isfile(
                        os.path.join(
                            root_path,
                            "labels",
                            file.replace(".png", "_labelTrainIds.png"),
                        )
                    ):
                        img_file = os.path.join(root_path, "images", "%s" % file)
                        label_file = os.path.join(
                            root_path,
                            "labels",
                            "%s" % file.replace(".png", "_labelTrainIds.png"),
                        )
                        init_list.append(
                            {"img": img_file, "name": file, "label": label_file}
                        )
                    else:
                        img_file = os.path.join(root_path, "images", "%s" % file)
                        init_list.append({"img": img_file, "name": file})

        dataset_dicts = []
        for i, entry in enumerate(init_list):
            record = {}
            record["file_name"] = entry["img"]
            record["image_id"] = f"generated_{i+1}"
            filename = os.path.join(entry["img"])
            if "label" in entry:
                record["sem_seg_file_name"] = entry["label"]
            width, height = Image.open(filename).size
            record["height"] = height
            record["width"] = width
            dataset_dicts.append(record)
        with open(os.path.join(root_path, "train_dict.json"), "w") as f:
            json.dump(dataset_dicts, f)
    return dataset_dicts


def get_datasets_from_directory(directory):
    return [
        d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))
    ]


def register_dataset(version, folder_name):
    DatasetCatalog.register(
        f"sd_{version}", lambda version=version: my_sd_dataset_function(folder_name)
    )
    MetadataCatalog.get(f"sd_{version}").set(
        stuff_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
    )
    MetadataCatalog.get(f"sd_{version}").set(
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
    MetadataCatalog.get(f"sd_{version}").set(
        thing_classes=[k["name"] for k in CITYSCAPES_CATEGORIES]
    )
    MetadataCatalog.get(f"sd_{version}").set(
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
    MetadataCatalog.get(f"sd_{version}").set(ignore_label=255)
    MetadataCatalog.get(f"sd_{version}").set(evaluator_type="sd_sem_seg")


directory = (
    "datasets/stable_diffusion"  # Change this to the path where all versions are stored
)
datasets = get_datasets_from_directory(directory)

for dataset in datasets:
    version = dataset.replace("sd_", "")
    register_dataset(version, dataset)
