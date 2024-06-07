FROM vault.habana.ai/gaudi-docker/1.15.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest

# Installs pdsh and upgrade pip
RUN apt-get update && apt-get install -y pdsh && \
   python -m pip install --upgrade pip

# Installs hccl_ofi_wrapper to interact with libfabric to utilize HW and networking mode (EFA)
ARG OFI_WRAPPER_WS="/root/hccl_ofi_wrapper"
RUN git clone "https://github.com/HabanaAI/hccl_ofi_wrapper.git" "${OFI_WRAPPER_WS}" && \
   cd "${OFI_WRAPPER_WS}" && \
   LIBFABRIC_ROOT=/opt/amazon/efa make

# Docker ssh port setup
RUN sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config && \
   sed -i 's/#   Port 22/    Port 3022/g' /etc/ssh/ssh_config && \
   sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
   service ssh restart

# Installs Optimum Habana and Habana's fork of DeepSpeed
RUN pip install optimum[habana] && \
   pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.15.0

CMD ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa && \
   chmod 600 ~/.ssh/id_rsa && \
   cat ~/.ssh/id_rsa.pub > ~/.ssh/authorized_keys && \
   /bin/bash
