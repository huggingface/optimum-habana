ARG BASE_IMAGE_NAME=vault.habana.ai/gaudi-docker/1.15.1/ubuntu22.04/habanalabs/pytorch-installer-2.2.0
ARG BASE_IMAGE_TAG=latest

FROM ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG} AS optimum-habana

ARG GAUDI_SW_VER=1.15.1
ARG OPTIMUM_HABANA_VER=1.11.1

RUN pip install --no-cache-dir optimum-habana==${OPTIMUM_HABANA_VER} && \
    pip install --no-cache-dir git+https://github.com/HabanaAI/DeepSpeed.git@${GAUDI_SW_VER}

FROM optimum-habana AS optimum-habana-examples

ARG OPTIMUM_HABANA_VER=1.11.1

WORKDIR /workspace

RUN git clone https://github.com/huggingface/optimum-habana.git --single-branch --branch v${OPTIMUM_HABANA_VER}

COPY requirements.txt .
RUN pip install -r requirements.txt
