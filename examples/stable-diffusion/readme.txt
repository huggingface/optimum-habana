This is experimental PR for Fal.ai ask based on current FLUX PR https://github.com/huggingface/optimum-habana/pull/1331
* PR is fixed with timing (HPU device sync and include VAD into timing measure)
* Added FP8 quantization support

To run sample with 1 image 1 batch in BF16 precision:
./run_bf16.sh

To run sample with 1 image 1 batch in FP8 precision (quant weights were tuned with 1 prompt):
./run_fp8.sh

To run sample with 1 image 1 batch in FP8 precision (quant weights were tuned with 500 prompts):
./run_fp8_500.sh
