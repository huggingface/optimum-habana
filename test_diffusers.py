from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana import GaudiConfig


gaudi_config = GaudiConfig(
    use_habana_mixed_precision=False,
)

generator = GaudiStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    gaudi_config=gaudi_config,
    use_lazy_mode=True,
    seed=27,
)

image = generator("An image of a squirrel in Picasso style").images[0]
image.save("image_of_squirrel_painting.png")
