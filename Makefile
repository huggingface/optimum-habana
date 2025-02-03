#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-habana.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))


.PHONY:	style test

# Run code quality checks
style_check: clean
	pip install -U pip ruff
	ruff check . setup.py
	ruff format --check . setup.py

style: clean
	pip install -U pip ruff
	ruff check . setup.py --fix
	ruff format . setup.py

# Run unit and integration tests
fast_tests:
	python -m pip install .[tests]
	python -m pytest tests/test_gaudi_configuration.py tests/test_trainer_distributed.py tests/test_trainer.py tests/test_trainer_seq2seq.py
# TODO enable when CI has more servers
#	python -m pytest test_functional_text_generation_example.py

# Run unit and integration tests related to Diffusers
fast_tests_diffusers:
	python -m pip install .[tests]
	python -m pip install -r examples/stable-diffusion/requirements.txt
	python -m pytest tests/test_diffusers.py

# Run single-card non-regression tests on image classification models
fast_tests_image_classifications:
	pip install timm
	python -m pip install .[tests]
	python -m pytest tests/test_image_classification.py

# Run unit and integration tests related to Image segmentation
fast_tests_image_segmentation:
	python -m pip install .[tests]
	python -m pytest tests/test_image_segmentation.py

# Run unit and integration tests related to text feature extraction
fast_tests_feature_extraction:
	python -m pip install .[tests]
	python -m pytest tests/test_feature_extraction.py

# Run unit and integration tests related to VideoMAE
fast_test_videomae:
	python -m pip install .[tests]
	python -m pytest tests/test_video_mae.py

# Run unit and integration tests related to Image segmentation
fast_tests_object_detection:
	python -m pip install .[tests]
	python -m pytest tests/test_object_detection.py

# Run integration tests related to table transformers
fast_tests_table_transformers:
	python -m pip install .[tests]
	python -m pytest tests/test_table_transformer.py

# Run non-performance regressions
slow_tests_custom_file_input: test_installs
	python -m pip install -r examples/language-modeling/requirements.txt
	python -m pytest tests/test_custom_file_input.py

# Run single-card non-regression tests
slow_tests_1x: test_installs
	python -m pytest tests/test_examples.py -v -s -k "single_card"
	python -m pip install peft==0.10.0
	python -m pytest tests/test_peft_inference.py
	python -m pytest tests/test_pipeline.py

# Run multi-card non-regression tests
slow_tests_8x: test_installs
	python -m pytest tests/test_examples.py -v -s -k "multi_card"

# Run DeepSpeed non-regression tests
slow_tests_deepspeed: test_installs
	python -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
	python -m pytest tests/test_examples.py -v -s -k "deepspeed"

slow_tests_diffusers: test_installs
	python -m pip install -r examples/stable-diffusion/requirements.txt
	python -m pytest tests/test_diffusers.py -v -s -k "textual_inversion"
	python -m pip install peft==0.7.0
	python -m pytest tests/test_diffusers.py -v -s -k "test_train_text_to_image_"
	python -m pytest tests/test_diffusers.py -v -s -k "test_train_controlnet"
	python -m pytest tests/test_diffusers.py -v -s -k "test_deterministic_image_generation"
	python -m pytest tests/test_diffusers.py -v -s -k "test_no_"

# Run text-generation non-regression tests
slow_tests_text_generation_example: test_installs
	python -m pip install triton==3.1.0 autoawq
	BUILD_CUDA_EXT=0 python -m pip install -vvv --no-build-isolation git+https://github.com/HabanaAI/AutoGPTQ.git
	python -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
	python -m pytest tests/test_text_generation_example.py tests/test_encoder_decoder.py -v -s --token $(TOKEN)

# Run image-to-text non-regression tests
slow_tests_image_to_text_example: test_installs
	python -m pytest tests/test_image_to_text_example.py -v -s --token $(TOKEN)

# Run visual question answering tests
slow_tests_openclip_vqa_example: test_installs
	python -m pip install -r examples/visual-question-answering/openclip_requirements.txt
	python -m pytest tests/test_openclip_vqa.py

slow_tests_fsdp: test_installs
	python -m pytest tests/test_fsdp_examples.py -v -s --token $(TOKEN)

slow_tests_trl: test_installs
	python -m pip install trl==0.9.6
	python -m pip install peft==0.12.0
	python -m pytest tests/test_trl.py -v -s -k "test_calculate_loss"

slow_tests_object_segmentation: test_installs
	python -m pytest tests/test_object_segmentation.py

# Check if examples are up to date with the Transformers library
example_diff_tests: test_installs
	python -m pytest tests/test_examples_match_transformers.py

# Utilities to release to PyPi
build_dist_install_tools:
	python -m pip install build
	python -m pip install twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*

build_doc_docker_image:
	docker build -t doc_maker --build-arg commit_sha=$(COMMIT_SHA_SUBPACKAGE) --build-arg clone_url=$(REAL_CLONE_URL) ./docs

doc: build_doc_docker_image
	@test -n "$(BUILD_DIR)" || (echo "BUILD_DIR is empty." ; exit 1)
	@test -n "$(VERSION)" || (echo "VERSION is empty." ; exit 1)
	docker run -v $(CURRENT_DIR):/doc_folder --workdir=/doc_folder doc_maker \
	doc-builder build optimum.habana /optimum-habana/docs/source/ \
		--repo_name optimum-habana \
		--build_dir $(BUILD_DIR) \
		--version $(VERSION) \
		--version_tag_suffix "" \
		--html \
		--clean

clean:
	find . -name "habana_log.livealloc.log_*" -type f -delete
	find . -name "hl-smi_log*" -type f -delete
	find . -name .lock -type f -delete
	find . -name .graph_dumps -type d -exec rm -r {} +
	find . -name save-hpu.pdb -type f -delete
	find . -name checkpoints.json -type f -delete
	find . -name hpu_profile -type d -exec rm -r {} +
	rm -rf regression/
	rm -rf tmp_trainer/
	rm -rf test/
	rm -rf build/
	rm -rf dist/
	rm -rf optimum_habana.egg-info/

test_installs:
	python -m pip install .[tests]
