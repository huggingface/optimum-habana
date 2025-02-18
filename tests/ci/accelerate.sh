# Description: Accelerate CI script

git clone https://github.com/huggingface/accelerate.git --branch hpu-support --depth 1 && cd accelerate && pip install -e .[testing] git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 make test_cli

if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_core

if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_fsdp

if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_deepspeed

if [ $? -ne 0 ]; then
    exit 1
fi

RUN_SLOW=1 PT_ENABLE_INT64_SUPPORT=1 PT_HPU_LAZY_MODE=0 make test_big_modeling

if [ $? -ne 0 ]; then
    exit 1
fi

#   - name: Run Accelerate Core tests
#     working-directory: ./accelerate
#     run: |
#       make test_core
#     env:
#       PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#       PT_HPU_LAZY_MODE: 0 # for fsdp
#       RUN_SLOW: 1

#   - name: Run Accelerate CLI tests
#     working-directory: ./accelerate
#     run: |
#       make test_cli
#     env:
#       PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#       RUN_SLOW: 1

#   - name: Run Accelerate FSDP tests
#     working-directory: ./accelerate
#     run: |
#       make test_fsdp
#     env:
#       PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#       PT_HPU_LAZY_MODE: 0 # for fsdp
#       RUN_SLOW: 1

#   - name: Run Accelerate DeepSpeed tests
#     working-directory: ./accelerate
#     run: |
#       make test_deepspeed
#     env:
#       PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#       PT_HPU_LAZY_MODE: 0 # for cpu offload
#       RUN_SLOW: 1

#   - name: Run Accelerate Big Modeling tests
#     working-directory: ./accelerate
#     run: |
#       make test_big_modeling
#     env:
#       PT_ENABLE_INT64_SUPPORT: 1 # for tokenizers
#       PT_HPU_LAZY_MODE: 0 # for cpu offload
#       RUN_SLOW: 1
