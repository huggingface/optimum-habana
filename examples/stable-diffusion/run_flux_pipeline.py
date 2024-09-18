import argparse
import torch
from optimum.habana.diffusers import GaudiFluxPipeline


parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=3, help="warmup iterations")
parser.add_argument("--iterations", type=int, default=3, help="warmup iterations")
parser.add_argument("--use_hpu_graph", action='store_true', help="use hpu graph")
args = parser.parse_args()


pipe = GaudiFluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
    use_habana=True,
    use_hpu_graphs=args.use_hpu_graph,
    gaudi_config="Habana/stable-diffusion",
)

if args.use_hpu_graph:
    from habana_frameworks.torch.hpu import wrap_in_hpu_graph
    pipe.transformer = wrap_in_hpu_graph(pipe.transformer)
prompt = "A cat in a bin, and holding a sign '/bin/cat' "
# Depending on the variant being used, the pipeline call will slightly vary.
# Refer to the pipeline documentation for more details.
print("warmuping...")
for i in range(args.warmup):
    image = pipe(prompt, num_inference_steps=28, guidance_scale=0.0).images[0]
torch.hpu.synchronize()
image = pipe(prompt, num_inference_steps=28, guidance_scale=0.0, profiling_warmup_steps=3, profiling_steps=3).images[0]
image.save("flux.png")
