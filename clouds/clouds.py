"""
# ---------------------------------------------------------------
# Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
# Licensed under the Apache License, Version 2.0

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py
           https://github.com/bytedance/fc-clip/blob/main/fcclip/fcclip.py

# ---------------------------------------------------------------
"""
from typing import Tuple
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from torch.nn.parallel import DistributedDataParallel

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
import copy
from my_utils import *
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import binary_erosion
from scipy.ndimage import label, sum as ndi_sum

from .sam import SAM

from .modeling.transformer_decoder.clouds_transformer_decoder import (
    MaskPooling,
    get_classification_logits,
)
from torch.nn.modules.dropout import _DropoutNd
from timm.models.layers import DropPath
import cv2


def is_element_in_string(my_list, my_string):
    for element in my_list:
        if element in my_string:
            return True
    return False


def show_anns(anns, val=0.35):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann["segmentation"]
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * val)))


def write_masks_to_png(
    masks,
    image,
    filename,
    path="segmented",
    val=0.35,
) -> None:
    plt.figure(figsize=(30, 30))
    plt.imshow(image)
    show_anns(masks, val)
    plt.axis("off")
    # plt.show()
    # filename = f"masks.png"
    plt.savefig(os.path.join(path, filename))
    return


#
# pred = processed_results[0]["sem_seg"].unsqueeze(dim=0)
# pred = torch.argmax(pred, dim=1)
# pred_1 = torch.squeeze(pred)
# pred_1 = np.asarray(pred_1.cpu().data, dtype=np.uint8)
# pred_1_map = colorize_mask(pred_1, None)

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(), model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


@META_ARCH_REGISTRY.register()
class CLOUDS(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # CLOUDS
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        classical_inference: bool,
        classical_inference_ema: bool,
        sam_enabled: bool,
        sam_mobile: bool,
        sam_minibatch: bool,
        sam_size_threshold: int,
        sam_erosion: bool,
        sam_erosion_size: int,
        sam_num_points: int,
        sam_selection_mode: str,
        sam_rm_intersection: bool,
        sam_refinement: bool,
        alpha_ema: float,
        overwriting: bool,
        iteration_update: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.sam_minibatch = sam_minibatch
        self.overwriting = overwriting
        if self.sam_minibatch:
            self.sem_seg_head_ema = deepcopy(self.sem_seg_head)
        self.local_iter = 0
        self.criterion = criterion
        self.num_queries = num_queries
        self.iteration_update = iteration_update
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer(
            "pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        # CLOUDS args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent)  # use this for void
        self.classical_inference = classical_inference
        self.classical_inference_ema = classical_inference_ema
        (
            _,
            self.train_num_templates,
            self.train_class_names,
        ) = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        (
            self.category_overlapping_mask,
            self.test_num_templates,
            self.test_class_names,
        ) = self.prepare_class_names_from_metadata(test_metadata, train_metadata)

        self.sam_enabled = sam_enabled
        if self.sam_enabled:
            self.sam = SAM(
                mobile=sam_mobile,
                size_threshold=sam_size_threshold,
                erosion=sam_erosion,
                erosion_size=sam_erosion_size,
                num_points=sam_num_points,
                selection_mode=sam_selection_mode,
                rm_intersection=sam_rm_intersection,
                refinement=sam_refinement,
            )

        self.sam_size_threshold = sam_size_threshold
        self.sam_erosion = sam_erosion
        self.sam_erosion_size = sam_erosion_size
        self.sam_num_points = sam_num_points
        self.sam_selection_mode = sam_selection_mode
        self.sam_rm_intersection = sam_rm_intersection
        self.sam_refinement = sam_refinement

        self.alpha_ema = alpha_ema

    def get_module(self, module):
        """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

        Args:
            module (MMDistributedDataParallel | nn.ModuleDict): The input
                module that needs processing.

        Returns:
            nn.ModuleDict: The ModuleDict of multiple networks.
        """
        if isinstance(module, DistributedDataParallel):
            return module.module

        return module

    def get_ema_model(self):
        return self.get_module(self.sem_seg_head_ema)

    def get_model(self):
        return self.get_module(self.sem_seg_head)

    def init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def update_ema_weights(self, iter):
        # alpha_teacher = min(1 - 1 / (iter + 1), self.alpha_ema)
        alpha_teacher = self.alpha_ema
        for ema_param, param in zip(
            self.get_ema_model().parameters(), self.get_model().parameters()
        ):
            if not param.data.shape:  # scalar tensor
                ema_param.data = (
                    alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
                )
            else:
                ema_param.data[:] = (
                    alpha_teacher * ema_param[:].data[:]
                    + (1 - alpha_teacher) * param[:].data[:]
                )

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(", ", ",")
                x_ = x_.split(",")  # there can be multiple synonyms for single class
                res.append(x_)
            return res

        # get text classifier
        try:
            class_names = split_labels(
                metadata.stuff_classes
            )  # it includes both thing and stuff
            train_class_names = split_labels(train_metadata.stuff_classes)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes)
            train_class_names = split_labels(train_metadata.thing_classes)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(
                set(test_class_names)
            )
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long
        )

        def fill_all_templates_ensemble(x_=""):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(
                templated_classes_num
            )  # how many templates for current classes
        class_names = templated_class_names
        # print("text for classification:", class_names)
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        (
            self.category_overlapping_mask,
            self.test_num_templates,
            self.test_class_names,
        ) = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(
                            self.train_class_names[idx : idx + bs], self.device
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT),
                    len(VILD_PROMPT),
                    text_classifier.shape[-1],
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(
                        self.backbone.get_text_classifier(
                            self.test_class_names[idx : idx + bs], self.device
                        ).detach()
                    )
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(
                    text_classifier.shape[0] // len(VILD_PROMPT),
                    len(VILD_PROMPT),
                    text_classifier.shape[-1],
                ).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.CLOUDS.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.CLOUDS.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.CLOUDS.ENSEMBLE_ON_VALID_MASK,
            "classical_inference": cfg.MODEL.CLOUDS.CLASSICAL_INFERENCE,
            "classical_inference_ema": cfg.MODEL.CLOUDS.CLASSICAL_INFERENCE_EMA,
            "sam_enabled": cfg.MODEL.CLOUDS.SAM.ENABLED,
            "sam_mobile": cfg.MODEL.CLOUDS.SAM.MOBILE,
            "sam_minibatch": cfg.MODEL.CLOUDS.SAM.MINIBATCH,
            "sam_size_threshold": cfg.MODEL.CLOUDS.SAM.SIZE_THRESHOLD,
            "sam_erosion": cfg.MODEL.CLOUDS.SAM.EROSION,
            "sam_erosion_size": cfg.MODEL.CLOUDS.SAM.EROSION_SIZE,
            "sam_num_points": cfg.MODEL.CLOUDS.SAM.NUM_POINTS,
            "sam_selection_mode": cfg.MODEL.CLOUDS.SAM.SELECTION_MODE,
            "sam_rm_intersection": cfg.MODEL.CLOUDS.SAM.RM_INTERSECTION,
            "sam_refinement": cfg.MODEL.CLOUDS.SAM.REFINEMENT,
            "alpha_ema": cfg.MODEL.CLOUDS.SAM.ALPHA_EMA,
            "overwriting": cfg.MODEL.CLOUDS.OVERWRITING,
            "iteration_update": cfg.MODEL.CLOUDS.ITERATION_UPDATE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        if self.training:
            if self.sam_minibatch:
                # Init/update ema model
                if self.local_iter == 0:
                    self.init_ema_weights()
                    # assert _params_equal(self.get_ema_model(), self.get_model())
                if not self.local_iter % self.iteration_update:
                    self.update_ema_weights(self.local_iter)
                    # assert not _params_equal(self.get_ema_model(), self.get_model())
                    # assert self.get_ema_model().training

        # We select the source images and augmented version of the generated ones
        images = [
            x["image_aug"].to(self.device)
            if "image_aug" in x
            else x["image"].to(self.device)
            for x in batched_inputs
        ]
        images_norm_list = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images_norm = ImageList.from_tensors(images_norm_list, self.size_divisibility)
        # We select the clean version of the generated ones
        images_clean = [
            x["image"].to(self.device) for x in batched_inputs if "image_aug" in x
        ]
        if images_clean:
            images_norm_list_clean = [
                (x - self.pixel_mean) / self.pixel_std for x in images_clean
            ]
            images_norm_clean = ImageList.from_tensors(
                images_norm_list_clean, self.size_divisibility
            )
            with torch.no_grad():
                features_clean = self.backbone(images_norm_clean.tensor)

        features = self.backbone(images_norm.tensor)

        text_classifier, num_templates = self.get_text_classifier()
        # Append void class weight
        text_classifier = torch.cat(
            [text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0
        )
        features["text_classifier"] = text_classifier
        features["num_templates"] = num_templates

        if images_clean:
            features_clean["text_classifier"] = text_classifier
            features_clean["num_templates"] = num_templates
        outputs = self.sem_seg_head(features)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances, images_norm)
            if images_clean:
                (
                    batched_inputs_target,
                    order_target,
                ) = separate_dicts_by_filename(batched_inputs)
                for m in self.get_ema_model().modules():
                    if isinstance(m, _DropoutNd):
                        m.training = False
                    if isinstance(m, DropPath):
                        m.training = False
                with torch.no_grad():
                    outputs_target = self.get_ema_model()(features_clean)
                    seg_maps_target = self.predict_inference(
                        outputs_target,
                        features_clean["clip_vis_dense"],
                        text_classifier,
                        num_templates,
                        images_norm_clean,
                        batched_inputs_target,
                    )
                    targets_target = process_segmentation_maps(seg_maps_target)
                    if self.sam_enabled:
                        separate_dict = separate_shapes_list(
                            targets_target, size_threshold=self.sam_size_threshold
                        )
                        coordinate_dict = get_fixed_points(
                            separate_dict,
                            apply_erosion=self.sam_erosion,
                            num_points=self.sam_num_points,
                            erosion_size=self.sam_erosion_size,
                            selection_mode=self.sam_selection_mode,
                        )
                        last_targets_target = []
                        for i, dico in enumerate(batched_inputs_target):
                            image_i = dico["image"]
                            image_perm = image_i.permute(1, 2, 0).cpu().numpy()
                            image_perm = self.sam.apply_image(image_perm)
                            self.sam.set_torch_image(
                                torch.tensor(image_perm.transpose(2, 0, 1))
                                .unsqueeze(0)
                                .to(self.device),
                                (768, 768),
                            )
                            points_coords, count_per_key = dict_to_tensor(
                                coordinate_dict[i]
                            )
                            points_coords = self.sam.apply_coords(
                                points_coords.cpu().numpy(), (768, 768)
                            )
                            if points_coords.shape[0]:
                                (masks, logits, masks_input,) = self.sam.predict_torch(
                                    point_coords=torch.tensor(points_coords).to(
                                        self.device
                                    ),
                                    point_labels=create_ones_tensor(points_coords).to(
                                        self.device
                                    ),
                                    multimask_output=True,
                                )

                                if self.sam_refinement:
                                    masks_input = select_best_masks(masks_input, logits)

                                    masks, logits, _, = self.sam.predict_torch(
                                        point_coords=torch.tensor(points_coords).to(
                                            self.device
                                        ),
                                        point_labels=create_ones_tensor(
                                            points_coords
                                        ).to(self.device),
                                        mask_input=masks_input.unsqueeze(1),
                                        multimask_output=True,
                                    )

                                masks = select_best_masks(masks, logits)
                                if self.sam_rm_intersection:
                                    masks = remove_intersecting_pixels(masks)

                                reconstructed_dict = reconstruct_dict(
                                    masks, count_per_key
                                )

                                new_targets_target = transform_masks(reconstructed_dict)
                                last_targets_target.append(new_targets_target)
                                viz_targets_target = union_of_masks(reconstructed_dict)
                                visualize_semantic_map_maxed(viz_targets_target)
                                save_semantic_map_maxed(viz_targets_target, after=True)
                            else:
                                last_targets_target.append(targets_target[i])

                        targets_target = last_targets_target
                for i, index in enumerate(order_target):
                    targets[index] = targets_target[i]

            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            self.local_iter += 1
            return losses

        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            if not self.classical_inference:
                # We ensemble the pred logits of in-vocab and out-vocab
                clip_feature = features["clip_vis_dense"]
                mask_for_pooling = F.interpolate(
                    mask_pred_results,
                    size=clip_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if "convnext" in self.backbone.model_name.lower():
                    pooled_clip_feature = self.mask_pooling(
                        clip_feature, mask_for_pooling
                    )
                    pooled_clip_feature = self.backbone.visual_prediction_forward(
                        pooled_clip_feature
                    )
                elif "rn" in self.backbone.model_name.lower():
                    pooled_clip_feature = self.backbone.visual_prediction_forward(
                        clip_feature, mask_for_pooling
                    )
                else:
                    raise NotImplementedError

                out_vocab_cls_results = get_classification_logits(
                    pooled_clip_feature,
                    text_classifier,
                    self.backbone.clip_model.logit_scale,
                    num_templates,
                )
                in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
                out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

                # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
                out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
                in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
                category_overlapping_mask = self.category_overlapping_mask.to(
                    self.device
                )

                if self.ensemble_on_valid_mask:
                    # Only include out_vocab cls results on masks with valid pixels
                    # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                    valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(
                        -1
                    ).sum(-1) > 0
                    valid_masking = valid_masking.to(
                        in_vocab_cls_results.dtype
                    ).unsqueeze(-1)
                    alpha = (
                        torch.ones_like(in_vocab_cls_results)
                        * self.geometric_ensemble_alpha
                    )
                    beta = (
                        torch.ones_like(in_vocab_cls_results)
                        * self.geometric_ensemble_beta
                    )
                    alpha = alpha * valid_masking
                    beta = beta * valid_masking
                else:
                    alpha = self.geometric_ensemble_alpha
                    beta = self.geometric_ensemble_beta

                cls_logits_seen = (
                    in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha
                ).log() * category_overlapping_mask
                cls_logits_unseen = (
                    in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta
                ).log() * (1 - category_overlapping_mask)
                cls_results = cls_logits_seen + cls_logits_unseen

                # This is used to filtering void predictions.
                is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
                mask_cls_probs = torch.cat(
                    [cls_results.softmax(-1) * (1.0 - is_void_prob), is_void_prob],
                    dim=-1,
                )
                mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images_norm.tensor.shape[-2], images_norm.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results,
                mask_pred_results,
                batched_inputs,
                images_norm.image_sizes,
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def predict_inference(
        self,
        outputs,
        clip_vis_dense,
        text_classifier,
        num_templates,
        images,
        batched_inputs,
    ):
        with torch.no_grad():
            # outputs = self.sem_seg_head(features)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            if self.classical_inference_ema:
                # We ensemble the pred logits of in-vocab and out-vocab
                clip_feature = clip_vis_dense
                mask_for_pooling = F.interpolate(
                    mask_pred_results,
                    size=clip_feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if "convnext" in self.backbone.model_name.lower():
                    pooled_clip_feature = self.mask_pooling(
                        clip_feature, mask_for_pooling
                    )
                    pooled_clip_feature = self.backbone.visual_prediction_forward(
                        pooled_clip_feature
                    )
                elif "rn" in self.backbone.model_name.lower():
                    pooled_clip_feature = self.backbone.visual_prediction_forward(
                        clip_feature, mask_for_pooling
                    )
                else:
                    raise NotImplementedError

                out_vocab_cls_results = get_classification_logits(
                    pooled_clip_feature,
                    text_classifier,
                    self.backbone.clip_model.logit_scale,
                    num_templates,
                )
                in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
                out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

                # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
                out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
                in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
                category_overlapping_mask = self.category_overlapping_mask.to(
                    self.device
                )

                if self.ensemble_on_valid_mask:
                    # Only include out_vocab cls results on masks with valid pixels
                    # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                    valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(
                        -1
                    ).sum(-1) > 0
                    valid_masking = valid_masking.to(
                        in_vocab_cls_results.dtype
                    ).unsqueeze(-1)
                    alpha = (
                        torch.ones_like(in_vocab_cls_results)
                        * self.geometric_ensemble_alpha
                    )
                    beta = (
                        torch.ones_like(in_vocab_cls_results)
                        * self.geometric_ensemble_beta
                    )
                    alpha = alpha * valid_masking
                    beta = beta * valid_masking
                else:
                    alpha = self.geometric_ensemble_alpha
                    beta = self.geometric_ensemble_beta

                cls_logits_seen = (
                    in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha
                ).log() * category_overlapping_mask
                cls_logits_unseen = (
                    in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta
                ).log() * (1 - category_overlapping_mask)
                cls_results = cls_logits_seen + cls_logits_unseen

                # This is used to filtering void predictions.
                is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
                mask_cls_probs = torch.cat(
                    [cls_results.softmax(-1) * (1.0 - is_void_prob), is_void_prob],
                    dim=-1,
                )
                mask_cls_results = torch.log(mask_cls_probs + 1e-8)

            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                r = retry_if_cuda_oom(self.semantic_inference)(
                    mask_cls_result, mask_pred_result
                )
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(
                        r, image_size, height, width
                    )
                processed_results[-1]["sem_seg"] = torch.argmax(r, dim=0)

            return processed_results

    def state_dict(
        self,
    ):
        # Get the default state dict
        modif_state_dict = super(CLOUDS, self).state_dict()
        if self.overwriting:
            # Exclude unwanted sub-modules
            unwanted_modules = ["sem_seg_head_ema.", "sam.sam."]
            unwanted_keys = [
                key
                for key in modif_state_dict.keys()
                if any(unwanted_module in key for unwanted_module in unwanted_modules)
            ]
            for key in unwanted_keys:
                del modif_state_dict[key]

        return modif_state_dict
