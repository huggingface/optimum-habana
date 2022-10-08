FROM vault.habana.ai/gaudi-docker/1.6.1/ubuntu20.04/habanalabs/pytorch-installer-1.12.0:latest

# Installs pdsh and upgrade pip
RUN apt-get update && apt-get install -y pdsh && \
   python -m pip install --upgrade pip

# Installs hccl_ofi_wrapper to interact with libfabric to utilize HW and networking mode (EFA)
ARG OFI_WRAPPER_WS="/root/hccl_ofi_wrapper"
RUN git clone "https://github.com/HabanaAI/hccl_ofi_wrapper.git" "${OFI_WRAPPER_WS}" && \
   cd "${OFI_WRAPPER_WS}" && \
   ln -s /opt/amazon/efa/lib64 /opt/amazon/efa/lib && \
   LIBFABRIC_ROOT=/opt/amazon/efa make

# Docker ssh port setup
RUN sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config && \
   sed -i 's/#   Port 22/    Port 3022/g' /etc/ssh/ssh_config && \
   sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
   service ssh restart

# Installs Optimum Habana and Habana's fork of DeepSpeed
ARG OPTIMUM_HABANA_WS="/root/optimum-habana"
RUN git clone https://github.com/huggingface/optimum-habana.git "${OPTIMUM_HABANA_WS}" && \
   cd "${OPTIMUM_HABANA_WS}" && git checkout multi_node && \
   pip install . \
   pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.6.1 && \
   cd examples/language-modeling && pip install -r requirements.txt

# Environment variables to communicate to all nodes when using DL1 instances
COPY .deepspeed_env "${OPTIMUM_HABANA_WS}"/examples/language-modeling

# List of all nodes with their IP address and number of devices
COPY hostfile "${OPTIMUM_HABANA_WS}"/examples/language-modeling

CMD ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa && \
   cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys && \
   /bin/bash
