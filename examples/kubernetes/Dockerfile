ARG GAUDI_SW_VER=1.16.2
ARG OS=ubuntu22.04
ARG TORCH_VER=2.2.2
ARG OPTIMUM_HABANA_VER=1.12.1

FROM vault.habana.ai/gaudi-docker/${GAUDI_SW_VER}/${OS}/habanalabs/pytorch-installer-${TORCH_VER}:latest AS optimum-habana

ARG GAUDI_SW_VER
ARG OPTIMUM_HABANA_VER

RUN pip install --no-cache-dir optimum-habana==${OPTIMUM_HABANA_VER} && \
    pip install --no-cache-dir git+https://github.com/HabanaAI/DeepSpeed.git@${GAUDI_SW_VER}

FROM optimum-habana AS optimum-habana-examples

ARG OPTIMUM_HABANA_VER

WORKDIR /workspace

RUN git clone https://github.com/huggingface/optimum-habana.git --single-branch --branch v${OPTIMUM_HABANA_VER}

COPY requirements.txt .
RUN pip install -r requirements.txt
