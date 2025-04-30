#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -xe

apt-get update
git clone https://github.com/amazon-science/mxeval.git
pip install -e mxeval
sed -i 's/npx tsc/tsc/g' mxeval/mxeval/execution.py
cp mbxp_evaluation/evaluation_setup/ubuntu.sh mxeval/language_setup/ubuntu.sh
PATH="$HOME/.rbenv/bin:$PATH" bash mxeval/language_setup/ubuntu.sh
