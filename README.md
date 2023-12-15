# Collaborating Foundation models for Domain Generalized Semantic Segmentation

## Overview

**Domain Generalized Semantic Segmentation** (DGSS)
deals with training a model on a labeled source domain
with the aim of generalizing to unseen domains during inference.
Existing DGSS methods typically effectuate robust
features by means of Domain Randomization (DR). Such an
approach is often limited as it can only account for style
diversification and not content. In this work, we take an
orthogonal approach to DGSS and propose to use an assembly of
**C**o**L**laborative F**OU**ndation models for **D**omain
Generalized **S**emantic Segmentation (**CLOUDS**). In detail,
**CLOUDS** is a framework that integrates FMs of various
kinds: (i) CLIP backbone for its robust feature represen-
tation, (ii) text-to-image generative models to diversify the
content, thereby covering various modes of the possible target
distribution, and (iii) Segment Anything Model (SAM)
for iteratively refining the predictions of the segmentation
model. Extensive experiments show that our CLOUDS excels in
adapting from synthetic to real DGSS benchmarks
and under varying weather conditions, notably outperforming
prior methods by 5.6% and 6.7% on averaged mIoU,
respectively.

<img src="imgs/main_figure.png" width="1000">
<div style="text-align: center;">
<img src="imgs/teaser.png" width="500">
</div>

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for CLOUDS](datasets/README.md).

See [Getting Started with CLOUDS](GETTING_STARTED.md).


## Checkpoints

We provide the following checkpoints for CLOUDS:

* [CLOUDS for GTA Domain Generalization](...)
* [CLOUDS for SYNTHIA Domain Generalization](...)
* [CLOUDS for Cityscapes Domain Generalization](...)


## Acknowledgement

[Mask2Former](https://github.com/facebookresearch/Mask2Former)

[FC-CLIP](https://github.com/bytedance/fc-clip)

[HRDA](https://github.com/lhoyer/HRDA)