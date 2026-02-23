#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -xe

MXEVAL_COMMIT="e09974f"

apt-get update
git clone https://github.com/amazon-science/mxeval.git
cd mxeval && git checkout ${MXEVAL_COMMIT} && cd ..
sed -i 's/evaluate_functional_correctness = mxeval.evaluate_functional_correctness"/evaluate_functional_correctness = mxeval.evaluate_functional_correctness:main"/' mxeval/setup.py
pip install -e mxeval --no-build-isolation
sed -i 's/npx tsc/tsc/g' mxeval/mxeval/execution.py
cp evaluation_setup/ubuntu.sh mxeval/language_setup/ubuntu.sh
PATH="$HOME/.rbenv/bin:$PATH" bash mxeval/language_setup/ubuntu.sh
