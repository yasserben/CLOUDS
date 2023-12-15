"""
Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
Licensed under the Apache License, Version 2.0

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""
from . import data  # register all new datasets
from . import modeling

# config
from .config import (
    add_maskformer2_config,
    add_clouds_config,
    add_wandb_config,
    add_prerocessing_training_set_config,
    add_repeat_factors,
)

# dataset loading
from .data.dataset_mappers.mapper_train import (
    MapperTrain
)
from .data.dataset_mappers.mapper_test import (
    MapperTest
)

# models
from .clouds import CLOUDS

# evaluation
from .evaluation.cityscapes_evaluation import CityscapesSemSegEvaluator
from .evaluation.semantic_evaluation import ClassicalSemSegEvaluator
from .engine.hooks import PersoEvalHook
