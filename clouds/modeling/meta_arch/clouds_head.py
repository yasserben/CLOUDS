"""
Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
Licensed under the Apache License, Version 2.0

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/meta_arch/mask_former_head.py
"""

import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.clouds_transformer_decoder import build_transformer_decoder
from ..transformer_decoder.mask2former_transformer_decoder import (
    build_original_transformer_decoder,
)
from ..transformer_decoder.clouds_bis_transformer_decoder import (
    build_bis_transformer_decoder,
)
from ..pixel_decoder.msdeformattn import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class CLOUDSHead(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        name_transformer_predictor: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes
        self.name_transformer_predictor = name_transformer_predictor

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            raise NotImplementedError
        if (
            cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
            == "MultiScaleMaskedTransformerDecoder"
        ):
            return {
                "input_shape": {
                    k: v
                    for k, v in input_shape.items()
                    if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
                },
                "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                "pixel_decoder": build_pixel_decoder(cfg, input_shape),
                "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
                "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
                "transformer_predictor": build_transformer_decoder(
                    cfg,
                    transformer_predictor_in_channels,
                    mask_classification=True,
                ),
                "name_transformer_predictor": cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME,
            }
        elif (
            cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
            == "OriginalMultiScaleMaskedTransformerDecoder"
        ):
            return {
                "input_shape": {
                    k: v
                    for k, v in input_shape.items()
                    if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
                },
                "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                "pixel_decoder": build_pixel_decoder(cfg, input_shape),
                "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
                "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
                "transformer_predictor": build_original_transformer_decoder(
                    cfg,
                    transformer_predictor_in_channels,
                    mask_classification=True,
                ),
                "name_transformer_predictor": cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME,
            }
        elif (
            cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME
            == "BisMultiScaleMaskedTransformerDecoder"
        ):
            return {
                "input_shape": {
                    k: v
                    for k, v in input_shape.items()
                    if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
                },
                "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                "pixel_decoder": build_pixel_decoder(cfg, input_shape),
                "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
                "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
                "transformer_predictor": build_bis_transformer_decoder(
                    cfg,
                    transformer_predictor_in_channels,
                    mask_classification=True,
                ),
                "name_transformer_predictor": cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME,
            }
        else:
            raise NotImplementedError

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        (
            mask_features,
            transformer_encoder_features,
            multi_scale_features,
        ) = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            if self.name_transformer_predictor == "MultiScaleMaskedTransformerDecoder":
                predictions = self.predictor(
                    multi_scale_features,
                    mask_features,
                    mask,
                    text_classifier=features["text_classifier"],
                    num_templates=features["num_templates"],
                )
            elif (
                self.name_transformer_predictor
                == "OriginalMultiScaleMaskedTransformerDecoder"
            ):
                predictions = self.predictor(multi_scale_features, mask_features, mask)
            elif (
                self.name_transformer_predictor
                == "BisMultiScaleMaskedTransformerDecoder"
            ):
                predictions = self.predictor(
                    multi_scale_features,
                    mask_features,
                    mask,
                    # text_classifier=features["text_classifier"],
                    # num_templates=features["num_templates"],
                )
        else:
            raise NotImplementedError
        return predictions
