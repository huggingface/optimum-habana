import json
from os import path


# Configuration Aux strings
class CFGS:
    ON = "on"
    OFF = "off"
    QUANTIZATION = "quantization"


# QuantConfig class
class QuantConfig:
    def __init__(self):
        self._quantization_enabled = False

    @property
    def quantization_enabled(self):
        return self._quantization_enabled

    @quantization_enabled.setter
    def quantization_enabled(self, val):
        self._quantization_enabled = val


def parse_quant_config(json_file_path: str) -> QuantConfig:
    quant_config = QuantConfig()
    assert path.isfile(json_file_path), "Quantization configuration file not found. Path - {}".format(json_file_path)
    with open(json_file_path, "r") as f:
        quant_cfg_json = json.load(f)
        if CFGS.QUANTIZATION in quant_cfg_json and quant_cfg_json[CFGS.QUANTIZATION] == CFGS.ON:
            quant_config.quantization_enabled = True

    return quant_config
