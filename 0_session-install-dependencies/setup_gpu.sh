#!/bin/bash

echo "installing GPU version of llama-cpp"

pip install llama-cpp-python==0.2.68 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
