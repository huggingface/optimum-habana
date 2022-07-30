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


.PHONY:	style test

# Run code quality checks
style_check:
	black --check --preview .
	isort --check .

style:
	black --preview .
	isort .

# Run unit and integration tests
fast_tests:
	pip install --user --no-cache-dir .[tests]
	python -m pytest tests/test_gaudi_configuration.py tests/test_trainer_distributed.py tests/test_trainer.py tests/test_trainer_seq2seq.py

# Run single-card non-regression tests
slow_tests_1x:
	pip install --user --no-cache-dir .[tests]
	python -m pytest tests/test_examples.py -v -s -k "single_card"

# Run multi-card non-regression tests
slow_tests_8x:
	pip install --user --no-cache-dir .[tests]
	python -m pytest tests/test_examples.py -v -s -k "multi_card"

# Check if examples are up to date with the Transformers library
example_diff_tests:
	pip install --user --no-cache-dir .[tests]
	python -m pytest tests/test_examples_match_transformers.py

# Utilities to release to PyPi
build_dist_install_tools:
	pip install --user build
	pip install --user twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*

build_doc_docker_image:
	docker build -t doc_maker ./docs

doc: build_doc_docker_image
	@test -n "$(BUILD_DIR)" || (echo "BUILD_DIR is empty." ; exit 1)
	@test -n "$(VERSION)" || (echo "VERSION is empty." ; exit 1)
	docker run -v $(CURRENT_DIR):/doc_folder --workdir=/doc_folder doc_maker \
	doc-builder build optimum.habana /optimum-habana/docs/source/ \
		--build_dir $(BUILD_DIR) \
		--version $(VERSION) \
		--html \
		--clean
