from optimum.habana.diffusers import GaudiDDIMScheduler
from pipeline_ddpm import GaudiDDPMPipeline
import logging
import sys
from optimum.habana.transformers.gaudi_configuration import GaudiConfig

model_name = "google/ddpm-ema-celebahq-256"
# # Setup logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
# logger.setLevel(logging.INFO)

scheduler = GaudiDDIMScheduler.from_pretrained(model_name) #, subfolder="scheduler")

gaudi_kwargs = {
    "use_torch_autocast": False,
}

gaudi_config = GaudiConfig(**gaudi_kwargs)

kwargs = {
    "scheduler": scheduler,
    "use_habana": True,
    "use_hpu_graphs": False,
    "gaudi_config": gaudi_config,
}

pipeline = GaudiDDPMPipeline.from_pretrained(model_name, **kwargs)

outputs = pipeline()
outputs.images[0].save('testing.jpg')
import pdb; pdb.set_trace()
