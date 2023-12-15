"""
Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
Licensed under the Apache License, Version 2.0
Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/mask_former_semantic_dataset_mapper.py
"""

import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
import matplotlib.pyplot as plt

from PIL import Image

__all__ = ["MapperTrain"]


class MapperTrain:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations_src,
        augmentations_sd,
        augmentations_photo,
        image_format,
        ignore_label,
        size_divisibility,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.tfm_gens_src = augmentations_src
        self.tfm_gens_sd = augmentations_sd
        self.tfm_gens_photometric = augmentations_photo
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(
            f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations_src}"
        )

    @classmethod
    def from_config(cls, cfg, is_train=True):
        augs_src = []
        augs_sd = []
        augs_photometric = []
        # Build augmentation
        if cfg.INPUT.RESIZE.ENABLED:
            augs_src.append(
                T.ResizeScale(
                    min_scale=0.5,
                    max_scale=2.0,
                    target_height=cfg.INPUT.INITIAL_HEIGHT,
                    target_width=cfg.INPUT.INITIAL_WIDTH,
                    interp=Image.BILINEAR,
                )
            )
        if cfg.INPUT.CROP.ENABLED:
            augs_src.append(
                T.FixedSizeCrop(
                    (768, 768),
                    pad=True,
                    seg_pad_value=255,
                    pad_value=0,
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs_src.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs_photometric.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
        if cfg.INPUT.FLIP:
            augs_src.append(T.RandomFlip())
            augs_sd.append(T.RandomFlip())

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations_src": augs_src,
            "augmentations_sd": augs_sd,
            "augmentations_photo": augs_photometric,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert (
            self.is_train
        ), "MaskFormerSemanticDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "sem_seg_file_name" in dataset_dict:
            # PyTorch transformation not implemented for uint16, so converting it to double first
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype(
                "double"
            )
        else:
            sem_seg_gt = np.full(
                (dataset_dict["height"], dataset_dict["width"]), self.ignore_label
            ).astype("double")

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                    dataset_dict["file_name"]
                )
            )

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        if not ("generated" in str(dataset_dict["image_id"])):
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens_src, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg
        else:
            aug_input, transforms = T.apply_transform_gens(self.tfm_gens_sd, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg
            aug_input_photo, transforms = T.apply_transform_gens(
                self.tfm_gens_photometric, aug_input
            )
            image_aug = aug_input_photo.image

        # Pad image and segmentation label here!
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if "generated" in str(dataset_dict["image_id"]):
            image_aug = torch.as_tensor(
                np.ascontiguousarray(image_aug.transpose(2, 0, 1))
            )
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if "generated" in str(dataset_dict["image_id"]):
                image_aug = F.pad(image_aug, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(
                    sem_seg_gt, padding_size, value=self.ignore_label
                ).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = image
        if "generated" in str(dataset_dict["image_id"]):
            dataset_dict["image_aug"] = image_aug

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError(
                "Semantic segmentation dataset should not have 'annotations'."
            )

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros(
                    (0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])
                )
            else:
                masks = BitMasks(
                    torch.stack(
                        [
                            torch.from_numpy(np.ascontiguousarray(x.copy()))
                            for x in masks
                        ]
                    )
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict
