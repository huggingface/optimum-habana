FROM vault.habana.ai/gaudi-docker/1.4.0/ubuntu20.04/habanalabs/pytorch-installer-1.10.2:1.4.0-442

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
RUN npm install npm@latest -g && \
    npm install n -g && \
    n latest

RUN git clone -b documentation https://github.com/huggingface/optimum-habana.git
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./optimum-habana[quality]
