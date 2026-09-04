#!/bin/bash
# Campaign 2: thin alias onto the slot scheduler (kind=bench, slots 3-4 only).
# Kept so campaign-1 scripts keep working. New work should call gpu_slot.sh --kind bench.
exec /home/jovyan/hackathon-shared/gpu_slot.sh --kind bench "$@"
