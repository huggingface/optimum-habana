from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiDDIMScheduler
from optimum.habana.transformers.trainer_utils import set_seed


model_name = "CompVis/stable-diffusion-v1-4"
scheduler = GaudiDDIMScheduler.from_config(model_name, subfolder="scheduler")

generator = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    scheduler=scheduler,
    safety_checker=None,
    use_habana=True,
    use_lazy_mode=False,
    use_hpu_graphs=True,
    gaudi_config="Habana/stable-diffusion",
)

set_seed(27)

outputs = generator(
    ["An image of a squirrel in Picasso style"],
    num_images_per_prompt=3,
    batch_size=1,
)

# generator.save_pretrained("my_pipeline")

for i, image in enumerate(outputs.images):
    image.save(f"image_{i+1}.png")
