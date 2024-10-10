This is experimental PR for Fal.ai ask based on current FLUX PR https://github.com/huggingface/optimum-habana/pull/1331
* PR is fixed with timing (HPU device sync and include VAD into timing measure)
* Added FP8 quantization support

To run sample with 1 image 1 batch in BF16 precision:
./run_bf16.sh

To run sample with 1 image 1 batch in FP8 precision (quant weights were tuned with 1 prompt):
./run_fp8.sh

To run sample with 1 image 1 batch in FP8 precision (quant weights were tuned with 500 prompts):
./run_fp8_500.sh

* Added batching
* Added --prompt_file option for large number of input prompts

To run sample with 5 prompts (batch size 1) in BF16 precision:
./run_bf16_prompts_5.sh

To run sample with 100 prompts (batch size 1) in BF16 precision:
./run_bf16_prompts_100.sh

* Added hybrid (mixed fp9 and bf16) precision denoising 

To run sample with 1 image 1 batch in hybrid precision:
./run_fp8_500_hybrid.sh

To run sample with 5 prompts (batch size 1) in hybrid precision:
./run_fp8_500_hybrid_prompts_5.sh
