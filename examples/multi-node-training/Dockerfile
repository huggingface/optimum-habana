FROM vault.habana.ai/gaudi-docker/1.6.1/ubuntu20.04/habanalabs/pytorch-installer-1.12.0:latest

RUN apt-get update && apt-get install -y pdsh && \
   python -m pip install --upgrade pip

 # Installs hccl_ofi_wrapper to interact with libfabric to utilize HW and networking mode (EFA)
 ARG OFI_WRAPPER_WS="/root/hccl_ofi_wrapper"
 RUN git clone "https://github.com/HabanaAI/hccl_ofi_wrapper.git" "${OFI_WRAPPER_WS}" && \
    cd "${OFI_WRAPPER_WS}" && \
    ln -s /opt/amazon/efa/lib64 /opt/amazon/efa/lib && \
    LIBFABRIC_ROOT=/opt/amazon/efa make

 RUN git clone https://github.com/huggingface/optimum-habana.git /root/optimum-habana && \
    cd /root/optimum-habana && \
    git checkout multi_node && \
    pip install . \
    pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.6.1 && \
    cd examples/language-modeling && pip install -r requirements.txt

COPY .deepspeed_env /root/optimum-habana/examples/language-modeling
