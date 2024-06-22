#!/bin/bash

pip install --upgrade --no-deps --force-reinstall -r 0_session-install-dependencies/requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"