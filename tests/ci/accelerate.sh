# Description: Accelerate CI script

hl-smi
echo "HABANA_VISIBLE_DEVICES=${HABANA_VISIBLE_DEVICES}"
echo "HABANA_VISIBLE_MODULES=${HABANA_VISIBLE_MODULES}"

git clone https://github.com/huggingface/accelerate.git --branch hpu-support --depth 1 && cd accelerate && pip install -e .[testing] git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 HABANA_VISIBLE_MODULES="" make test_cli
#       PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 HABANA_VISIBLE_MODULES="" make test_core
#      PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#      PT_HPU_LAZY_MODE: 0 # for fsdp
if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_fsdp
#      PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#      PT_HPU_LAZY_MODE: 0 # for fsdp
if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_deepspeed
#      PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#      PT_HPU_LAZY_MODE: 0 # for cpu offload
if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_big_modeling
#      PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#      PT_HPU_LAZY_MODE: 0 # for cpu offload
if [ $? -ne 0 ]; then
    exit 1
fi
