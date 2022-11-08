from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiPNDMScheduler
from optimum.habana.transformers.trainer_utils import set_seed


set_seed(27)

gaudi_config = GaudiConfig(
    use_habana_mixed_precision=False,
)

model_name = "CompVis/stable-diffusion-v1-4"
scheduler = GaudiPNDMScheduler.from_config(model_name, subfolder="scheduler")

generator = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    use_habana=True,
    use_lazy_mode=False,
    gaudi_config=gaudi_config,
    scheduler=scheduler,
)

outputs = generator(
    ["An image of a squirrel in Picasso style", "Sunset in Hollywood"],
    num_images_per_prompt=4,
    batch_size=1,
    num_inference_steps=50,
    height=512,
    width=512,
)

for i, image in enumerate(outputs.images):
    image.save(f"image_{i+1}.png")
