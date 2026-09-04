#!/bin/bash
# Campaign 2: thin alias onto the slot scheduler (kind=train, first free slot, 0 first).
# Kept so campaign-1 scripts keep working. New work should call gpu_slot.sh directly.
exec /home/jovyan/hackathon-shared/gpu_slot.sh --kind train "$@"
