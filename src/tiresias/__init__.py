#!/usr/bin/env python3

"""
Tiresias: a character-level, span-masked, encoder-only model

- re-exports the config, model, and data-collation classes
- registers Tiresias with the transformers Auto* factories on import so that a saved model can be reloaded with HF's ``AutoModelForMaskedLM.from_pretrained``
"""

from .modeling.configuration_tiresias import TiresiasConfig
from .modeling.modeling_tiresias import (
    CharEncoderLayer,
    CharHashEmbedding,
    TiresiasForMaskedLM,
    TiresiasMLMHead,
    TiresiasModel,
    TiresiasPreTrainedModel,
    create_local_attention_mask,
)
from .modeling.encoder import DeepEncoder, DeepEncoderLayer, DeepFeedForward, GeGLU
from .modeling.disentangled_attn import (
    DisentangledSelfAttention,
    build_relative_position,
    make_log_bucket_position,
)
from .modeling.dense_gbst import DenseGBST
from .data.data_collator_mlm import (
    DataCollatorForMaskedLM,
    DatasetForMaskedLM,
    MaskingScheduleType,
)

__all__ = [
    # config
    "TiresiasConfig",
    # model
    "TiresiasModel",
    "TiresiasForMaskedLM",
    "TiresiasPreTrainedModel",
    "TiresiasMLMHead",
    "CharEncoderLayer",
    "CharHashEmbedding",
    "create_local_attention_mask",
    # encoder components
    "DeepEncoder",
    "DeepEncoderLayer",
    "DeepFeedForward",
    "GeGLU",
    "DisentangledSelfAttention",
    "build_relative_position",
    "make_log_bucket_position",
    # downsampling
    "DenseGBST",
    # data utilities
    "DataCollatorForMaskedLM",
    "DatasetForMaskedLM",
    "MaskingScheduleType",
]

__version__ = "1.0.0"


def _register_auto_classes() -> None:
    """
    Register Tiresias w/ HF transformers Auto

    Two independent registrations:

    1. Local registration via ``AutoConfig/AutoModel/AutoModelForMaskedLM.register(...)`` to let  ``AutoModelForMaskedLM.from_pretrained("./your-dir")`` resolve Tiresias classes in current Pythonprocess w/out passing ``trust_remote_code=True``

    2. ``register_for_auto_class(...)`` on each class to tags classes so a later ``save_pretrained`` writes an ``auto_map`` field into ``config.json`` and copies the modeling source files into the saved dir. This allows users to load the model straight from local or Hub with ``trust_remote_code=True`` w/out installing the pkg
    """
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    try:
        AutoConfig.register("tiresias", TiresiasConfig)
        AutoModel.register(TiresiasConfig, TiresiasModel)
        AutoModelForMaskedLM.register(TiresiasConfig, TiresiasForMaskedLM)
    except ValueError:
        # in case tiresias model type already registered (safe to ignore)
        pass

    TiresiasConfig.register_for_auto_class()
    TiresiasModel.register_for_auto_class("AutoModel")
    TiresiasForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")

_register_auto_classes()