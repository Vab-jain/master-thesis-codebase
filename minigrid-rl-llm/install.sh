#!/bin/bash
set -e
echo "Starting install script..." >&2

# Remove apt commands; not available in this container
# apt update
# apt clean

# Upgrade pip and install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/stanfordnlp/dspy.git
pip install --upgrade "pydantic>=2.1.0"
pip install "numpy<2"