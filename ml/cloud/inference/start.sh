#!/bin/sh
set -eu
exec python -m holospex_ml.live --host 0.0.0.0 --port "${PORT:-8080}" --device cpu
