#  Copyright 2023 The HuggingFace Team. All rights reserved.
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

install-poetry:
	curl -sSL https://install.python-poetry.org | python3 -

gen-server:
	# Compile protos
	pip install grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir text_generation_server/pb || true
	python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

update-lock:
	rm poetry.lock
	poetry lock --no-update

export-requirements:
	poetry export -f requirements.txt --without-hashes --output requirements.txt
