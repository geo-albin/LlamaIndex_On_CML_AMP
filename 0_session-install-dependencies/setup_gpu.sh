#!/bin/bash

echo "installing GPU version of llama-cpp"
pip install llama-cpp-python numpy==1.26.4 --upgrade --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
