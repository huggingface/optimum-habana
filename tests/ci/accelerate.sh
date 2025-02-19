# Description: Accelerate CI script

hl-smi
echo "HABANA_VISIBLE_DEVICES=${HABANA_VISIBLE_DEVICES}"
echo "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES}"

# Install Accelerate and DeepSpeed
git clone https://github.com/huggingface/accelerate.git
cd accelerate
git checkout hpu-support
git pull
pip install -e .[testing]
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0

# Install Rust and build Safetensors
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
rustup update
git clone https://github.com/huggingface/safetensors
cd safetensors
git checkout fa833511664338bfc927fc02653ddb7d38d40be9
pip install setuptools_rust
pip install -e bindings/python
cd ..

# Set environment variables
export PT_ENABLE_INT64_SUPPORT=1
export PT_HPU_LAZY_MODE=0
export RUN_SLOW=1

fuser -k 29500/tcp

echo "Running CLI tests"
make test_cli
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Running Big Modeling tests"
make test_big_modeling
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Running Core tests"
make test_core
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Running FSDP tests"
make test_fsdp
if [ $? -ne 0 ]; then
    exit 1
fi

echo "Running DeepSpeed tests"
make test_deepspeed
if [ $? -ne 0 ]; then
    exit 1
fi
