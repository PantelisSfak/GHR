# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Model class."""

import warnings
from collections import OrderedDict

from transformers.utils import logging
from adapter_transformers.src.transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from adapter_transformers.src.transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

from adapter_transformers.src.transformers.models.electra.modeling_electra import ElectraForQuestionAnswering

logger = logging.get_logger(__name__)


MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping

        ("electra", "ElectraModel"),
  
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping

        ("electra", "ElectraForPreTraining"),
 
    ]
)



MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping

        ("electra", "ElectraForQuestionAnswering"),

    ]
)



MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)

MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.

class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


