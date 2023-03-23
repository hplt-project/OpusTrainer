#!/bin/bash
set -euo pipefail

export PYTHONPATH="src:${PYTHONPATH:-}"
python3 -m unittest discover -s tests