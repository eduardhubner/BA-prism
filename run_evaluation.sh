#!/bin/bash
# Load environment variables from .env file
set -a
source .env
set +a

# Run evaluation
python src/evaluation.py
