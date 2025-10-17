# Parameterize base image components
ARG DOCKER_URL=vault.habana.ai/gaudi-docker
ARG VERSION=1.22.1
ARG BASE_NAME=ubuntu24.04
ARG PT_VERSION=2.7.1
ARG REVISION=latest
ARG REPO_TYPE=habanalabs

FROM ${DOCKER_URL}/${VERSION}/${BASE_NAME}/${REPO_TYPE}/pytorch-installer-${PT_VERSION}:${REVISION}
# Parameterize commit/branch for optimum-habana and DeepSpeed checkout
ARG OH_FORK_COMMIT=v1.19.0
ARG DEEPSPEED_COMMIT=1.22.0

ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

RUN apt update && \
    apt install -y gettext moreutils jq && \
    apt install -y npm && \
    npm install n -g && \
    ln -sf /usr/bin/python3 /usr/bin/python

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@${DEEPSPEED_COMMIT}

WORKDIR /root

RUN git clone https://github.com/huggingface/optimum-habana && \
    cd optimum-habana && \
    git checkout ${OH_FORK_COMMIT} && \
    pip install -e .

WORKDIR /root/optimum-habana/examples/text-generation
RUN python3 -m pip install -r requirements.txt && \
    python3 -m pip install -r requirements_lm_eval.txt 
