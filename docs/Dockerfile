FROM vault.habana.ai/gaudi-docker/1.4.0/ubuntu20.04/habanalabs/pytorch-installer-1.10.2:1.4.0-442

RUN git clone -b documentation https://github.com/huggingface/optimum-habana.git
RUN pip install --upgrade pip

WORKDIR optimum-habana/

RUN pip install .[quality]
