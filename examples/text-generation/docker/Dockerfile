# Parameterize base image components
ARG DOCKER_URL=vault.habana.ai/gaudi-docker
ARG VERSION=1.23.0
ARG BASE_NAME=ubuntu24.04
ARG PT_VERSION=2.9.0
ARG REVISION=latest
ARG REPO_TYPE=habanalabs

FROM ${DOCKER_URL}/${VERSION}/${BASE_NAME}/${REPO_TYPE}/pytorch-installer-${PT_VERSION}:${REVISION}
# Parameterize commit/branch for vllm+plugin checkout
ARG OH_COMMIT=v1.20.0
ARG OH_PLUGIN_HASH=db9688c

# BUILD_ARGS="--build-arg http_proxy --build-arg https_proxy --build-arg no_proxy"
# CPREFIX=gar-registry.caas.intel.com/aice/
# CNAME=pyt-${PT_VERSION}-${VERSION}-rel-ub24-vllm-${OH_COMMIT}-${OH_PLUGIN_HASH}
# docker pull ${DOCKER_URL}/${VERSION}/${BASE_NAME}/${REPO_TYPE}/pytorch-installer-${PT_VERSION}:${REVISION}
# docker build -f Dockerfile-${VERSION}-rel-ub24-oh-${OH_COMMIT}-${OH_PLUGIN_HASH} -t ${CPREFIX}${CNAME} $BUILD_ARGS .

ENV OMPI_MCA_btl_vader_single_copy_mechanism=none

RUN apt update && \
    apt install -y gettext moreutils jq && \
    ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /root

# Install vllm inside the container
ENV VLLM_TARGET_DEVICE=hpu

RUN git clone https://github.com/huggingface/optimum-habana && \
    cd optimum-habana && \
    git checkout v1.20.0 && \
    pip install -v -e .

RUN git clone https://github.com/HabanaAI/vllm-fork.git && \
    cd vllm-fork && \
    pip install -v -e .
    #pip install -v -e . && \
#    pip install setuptools==69.5.1

## Install packages required for AICE automation
RUN pip install datasets &&\
    pip install pandas && \
    pip install lm-eval &&\
    pip install lm-eval[api] &&\
    pip install matplotlib &&\
    pip install langdetect &&\
    pip install immutabledict && \
    pip install streamlit
    
CMD ["/bin/bash", "-c", "service ssh restart; sleep infinity"]

