# Clone the original TGI repo
FROM python:3.9 as tgi
WORKDIR /tmp
RUN git clone --depth 1 --branch v0.9.2 https://github.com/huggingface/text-generation-inference.git

# Rust builder
FROM lukemathwalker/cargo-chef:latest-rust-1.70 AS chef
WORKDIR /usr/src

FROM chef as planner
COPY --from=tgi /tmp/text-generation-inference/Cargo.toml Cargo.toml
COPY --from=tgi /tmp/text-generation-inference/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tmp/text-generation-inference/proto proto
COPY --from=tgi /tmp/text-generation-inference/benchmark benchmark
COPY --from=tgi /tmp/text-generation-inference/router router
COPY --from=tgi /tmp/text-generation-inference/launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY --from=tgi /tmp/text-generation-inference/Cargo.toml Cargo.toml
COPY --from=tgi /tmp/text-generation-inference/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tmp/text-generation-inference/proto proto
COPY --from=tgi /tmp/text-generation-inference/benchmark benchmark
COPY --from=tgi /tmp/text-generation-inference/router router
COPY --from=tgi /tmp/text-generation-inference/launcher launcher
RUN cargo build --release

# Text Generation Inference base image
FROM vault.habana.ai/gaudi-docker/1.11.0/ubuntu22.04/habanalabs/pytorch-installer-2.0.1:latest as base

# Text Generation Inference base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

# libssl.so.1.1 is not installed on Ubuntu 22.04 by default, install it
RUN wget http://nz2.archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb && \
    dpkg -i ./libssl1.1_1.1.1f-1ubuntu2_amd64.deb

WORKDIR /usr/src

COPY --from=tgi /tmp/text-generation-inference/proto proto
COPY --from=tgi /tmp/text-generation-inference/server server
COPY --from=tgi /tmp/text-generation-inference/server/Makefile server/Makefile
# Copy files modified for running on Gaudi
COPY server/text_generation_server/cli.py server/text_generation_server/cli.py
COPY server/text_generation_server/server.py server/text_generation_server/server.py
COPY server/text_generation_server/models/__init__.py server/text_generation_server/models/__init__.py
COPY server/text_generation_server/models/bloom.py server/text_generation_server/models/bloom.py
COPY server/text_generation_server/models/causal_lm.py server/text_generation_server/models/causal_lm.py
COPY server/text_generation_server/models/model.py server/text_generation_server/models/model.py
COPY server/text_generation_server/models/santacoder.py server/text_generation_server/models/santacoder.py
COPY server/text_generation_server/utils/watermark.py server/text_generation_server/utils/watermark.py
COPY server/text_generation_server/utils/tokens.py server/text_generation_server/utils/tokens.py
COPY server/text_generation_server/utils/logits_process.py server/text_generation_server/utils/logits_process.py
COPY server/text_generation_server/utils/dist.py server/text_generation_server/utils/dist.py
COPY server/requirements.txt server/requirements.txt
COPY server/pyproject.toml server/pyproject.toml
# Install server
RUN cd server && \
    make gen-server && \
    pip install -r requirements.txt && \
    pip install . --no-cache-dir

# Install benchmarker
COPY --from=builder /usr/src/target/release/text-generation-benchmark /usr/local/bin/text-generation-benchmark
# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher

# Final image
FROM base

ENTRYPOINT ["text-generation-launcher"]
CMD ["--json-output"]
