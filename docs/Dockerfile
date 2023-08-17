FROM vault.habana.ai/gaudi-docker/1.11.0/ubuntu20.04/habanalabs/pytorch-installer-2.0.1:latest

ARG commit_sha
ARG clone_url

# Need node to build doc HTML. Taken from https://stackoverflow.com/a/67491580
RUN apt-get update && apt-get install -y \
    software-properties-common \
    npm
# Needs nodejs >= v14.x but Ubuntu 22 repos are stuck to v12.x
RUN apt-get remove -y libnode-dev &&\
    apt-get autoremove -y &&\
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - &&\
    apt-get install -y nodejs
RUN npm install npm@latest -g && \
    npm install n -g && \
    n latest

RUN git clone $clone_url && cd optimum-habana && git checkout $commit_sha
RUN python3 -m pip install --no-cache-dir --upgrade pip
RUN python3 -m pip install --no-cache-dir ./optimum-habana[quality]
