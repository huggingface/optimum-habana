FROM vault.habana.ai/gaudi-docker/1.13.0/ubuntu20.04/habanalabs/pytorch-installer-2.1.0:latest

ARG commit_sha
ARG clone_url

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
RUN npm install npm@9.8.1 -g && \
    npm install n -g && \
    n latest

RUN git clone $clone_url optimum-habana && cd optimum-habana && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./optimum-habana[quality]
