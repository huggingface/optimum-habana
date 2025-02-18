# Description: Accelerate CI script

git clone https://github.com/huggingface/accelerate.git --branch hpu-support --depth 1 && cd accelerate
pip install -e .[testing] git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 make test_cli
RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_core
RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_fsdp
RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_deepspeed
RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_big_modeling
