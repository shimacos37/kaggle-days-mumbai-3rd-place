#!/bin/bash
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids-0.18
exec "$@"
