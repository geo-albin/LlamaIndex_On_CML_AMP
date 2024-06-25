#!/bin/bash

echo "installing GPU version of llama-cpp"

# pip install llama-cpp-python==0.2.79 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# pip install llama-cpp-python==0.2.79 --upgrade --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122


# pip install llama-cpp-python==0.2.68 numpy==1.26.4 pandas==2.1.4 --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

pip install numpy==1.26.4 pandas==2.1.4 llama-cpp-python==0.2.74 --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122


