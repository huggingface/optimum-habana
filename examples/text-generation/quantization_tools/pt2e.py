import os
from dataclasses import dataclass

import torch


# Dictionary for quantization data types
qdtype_dict = {"int8": torch.int8, "fp8_143": torch.float8_e4m3fn, "fp8_152": torch.float8_e5m2}


@dataclass
class PT2EConfig:
    qdtype: torch.dtype
    save: bool
    model_path: str
    logger: any


def pt2e_prepare(model, qdtype_key, save, path, logger):
    """
    This function initializes the model with the PT2E configuration and either returns the model for calibration
    or loads an already saved model from the given path and returns it.
    """
    # Initialize the model's PT2E configuration
    model.pt2e_config = PT2EConfig(qdtype=qdtype_dict[qdtype_key], save=save, model_path=path, logger=logger)

    config = model.pt2e_config
    if model.config.model_type != "llama":
        return model

    import habana_frameworks.torch.core as htcore

    htcore.hpu_inference_initialize(model, mark_non_scales=False)

    from habana_frameworks.torch.core.quantizer import (
        habana_quant_config_symmetric,
        habana_quantizer,
    )
    from torch.ao.quantization.quantize_pt2e import prepare_pt2e
    from torch.export import export_for_training

    if config.save:
        # Export --> prepare_pt2e --> return model for calibration
        config.logger.info("[pt2e_quant] Using PT2 Export for calibration")
        quantizer = habana_quantizer()
        quant_config = habana_quant_config_symmetric(config.qdtype)
        quantizer.set_global(quant_config)
        exported_model = export_for_training(model.model)
        if isinstance(exported_model, torch.export.exported_program.ExportedProgram):
            exported_model = exported_model.module()
        config.logger.info("[pt2e_quant] Inserting observers for measurement")
        model.model = prepare_pt2e(exported_model, quantizer)
        return model
    else:
        # Load model with quantization info --> return model for inference
        load_path = (
            config.model_path + "pt2e_quant_model.pt2" if os.path.isdir(config.model_path) else config.model_path
        )
        config.logger.info(f"[pt2e_quant] Using PT2 Export load from {load_path}")
        del model.model
        model.model = torch.export.load(load_path).module()
        config.logger.info("[pt2e_quant] Loading done!")
        return model


def pt2e_save(model):
    """
    This function calls converts_pt2e after model calibration and followed by torch.export.save.
    """
    assert hasattr(model, "pt2e_config"), "Please call pt2e_prepare and run calibration before calling pt2e_save."
    config = model.pt2e_config
    from torch.ao.quantization.quantize_pt2e import convert_pt2e

    config.logger.info("[pt2e_quant] Converting model after calibration")
    model.model = convert_pt2e(model.model)
    save_path = config.model_path + "pt2e_quant_model.pt2" if os.path.isdir(config.model_path) else config.model_path
    config.logger.info(f"[pt2e_quant] Using PT2 Export save at {save_path}")
    with torch.no_grad():
        torch.export.save(model.model, save_path)
        config.logger.info("[pt2e_quant] Saving done!")
