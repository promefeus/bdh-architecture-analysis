from configs.config import ModelConfig

from models.transformer import Transformer
from models.bdh_base import BDH as BDHBase
from models.bdh_nomul import BDH as BDHNoMul
from models.bdh_lowdim import BDH as BDHLowDim
from models.bdh_improved import BDH as BDHImproved


def get_model(name):
    config = ModelConfig()

    if name == "transformer":
        return Transformer(config)

    elif name == "bdh_base":
        return BDHBase(config)

    elif name == "bdh_nomul":
        return BDHNoMul(config)

    elif name == "bdh_lowdim":
        return BDHLowDim(config)

    elif name == "bdh_improved":
        return BDHImproved(config)

    else:
        raise ValueError(f"Unknown model: {name}")