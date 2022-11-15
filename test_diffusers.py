from optimum.habana import GaudiConfig
from optimum.habana.diffusers import GaudiStableDiffusionPipeline
from optimum.habana.diffusers.schedulers import GaudiDDIMScheduler, GaudiPNDMScheduler
from optimum.habana.transformers.trainer_utils import set_seed


set_seed(27)

gaudi_config = GaudiConfig(
    use_habana_mixed_precision=False,
)

model_name = "CompVis/stable-diffusion-v1-4"
scheduler = GaudiDDIMScheduler.from_config(model_name, subfolder="scheduler")

lazy_mode = True
hpu_graphs = False
print(f"Lazy mode: {lazy_mode}; HPU graphs: {hpu_graphs}")

generator = GaudiStableDiffusionPipeline.from_pretrained(
    model_name,
    use_habana=True,
    use_lazy_mode=lazy_mode,
    use_hpu_graphs=hpu_graphs,
    gaudi_config=gaudi_config,
    scheduler=scheduler,
)

outputs = generator(
    ["An image of a squirrel in Picasso style"],
    num_images_per_prompt=64,
    batch_size=2,
    num_inference_steps=4,
    height=16,
    width=16,
)

# for i, image in enumerate(outputs.images):
#     image.save(f"image_{i+1}.png")
