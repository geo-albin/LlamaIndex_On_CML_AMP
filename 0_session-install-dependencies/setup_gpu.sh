#!/bin/bash

echo "installing GPU version of llama-cpp"

pip install numpy==1.26.4 pandas==2.1.4 llama-cpp-python==0.2.82 --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install numpy==1.26.4 pandas==2.1.4 llama-cpp-python==0.2.82 --force-reinstall --no-cache-di

