FROM vault.habana.ai/gaudi-docker/1.19.0/ubuntu22.04/habanalabs/pytorch-installer-2.5.1:latest

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
RUN npm install n -g && \
    n latest

RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --upgrade-strategy eager optimum[habana]
RUN python3 -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0

RUN mkdir -p /workspace
WORKDIR /workspace
RUN git clone https://github.com/huggingface/optimum-habana && cd optimum-habana && git checkout v1.15.0

WORKDIR /workspace/optimum-habana/examples/text-generation
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r requirements_lm_eval.txt

