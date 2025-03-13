import subprocess
import os

#output, input, bs, dtype
PARAM_TO_TEST = [
    (4224, 128, 202, "fp8"),
    (2176, 2048, 115, "fp8"),
    (256, 128, 1664, "fp8"), #fail - Synapse is being terminated due to fatality
    (2176, 128, 373, "fp8"),
    (4096, 2048, 87, "fp8"),
    (129, 128, 1, "fp8"),
    (4096, 2048, 109, "fp8"),
    (2049, 2048, 1, "fp8"), #fail - Synapse is being terminated due to fatality
    (2176, 2048, 128, "fp8"),
    (4096, 2048, 150, "fp8"), #oom
    (2176, 128, 256, "fp8"),
    (2176, 2048, 128, "fp8"),
    (256, 128, 1920, "fp8"),
    (4224, 128, 256, "fp8"),
    (2176, 128, 512, "fp8"),  #device critical error after generation
    (4224, 128, 128, "bf16"),
    (256, 128, 1152, "bf16"),
    (2176, 2048, 64, "bf16"),
    (129, 128, 1, "bf16"),
    (4096, 2048, 64, "bf16"),
    (2176, 128, 256, "bf16"),
    (2049, 2048, 1, "bf16"),
    (2176, 2048, 152, "bf16"), #oom
    (4096, 2048, 116, "bf16"), #oom
    (256, 128, 1920, "bf16"), #oom
    (2176, 128, 272, "bf16"),
    (4224, 128, 125, "bf16"),
]


#TQDM_DISABLE=1 QUANT_CONFIG=quantization_config/maxabs_quant_mixtral.json  python3 ../gaudi_spawn.py --use_deepspeed --world_size 2 run_generation.py --model_name_or_path mistralai/Mixtral-8x7B-v0.1 --bf16 --use_kv_cache --use_hpu_graphs --limit_hpu_graphs --batch_size 1920 --max_new_tokens 128  --max_input_tokens 128    --reuse_cache --bucket_size 128 --bucket_internal
for output, input, bs, dtype in PARAM_TO_TEST:
    env_variables = os.environ.copy()
    env_variables["TQDM_DISABLE"] = "1"
    if dtype == "fp8":
        env_variables["QUANT_CONFIG"] = "quantization_config/maxabs_quant_mixtral.json"

    # you may need to put "--token", "<your token>"
    # if need, give full path
    command = ["python3", "../gaudi_spawn.py",
               "--use_deepspeed", "--world_size", "2",
               "run_generation.py",
               "--model_name_or_path", "mistralai/Mixtral-8x7B-v0.1",
               "--bf16", "--use_kv_cache", "--use_hpu_graphs", "--limit_hpu_graphs",
               "--reuse_cache", "--bucket_size", "128", "--bucket_internal",
               f"--batch_size={bs}", f"--max_new_tokens={output}", f"--max_input_tokens={input}"]

    print(f"\n{dtype} Command to test: {' '.join(command)}\n")
    proc = subprocess.run(command, env=env_variables)
    try:
        assert proc.returncode == 0
    except AssertionError as e:
        print(dtype, command, "failed to run")