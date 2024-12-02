#!/bin/bash

echo "Visible devices: $HABANA_VISIBLE_DEVICES"
hl-smi
python -m pip install --upgrade pip
make fast_tests_diffusers
