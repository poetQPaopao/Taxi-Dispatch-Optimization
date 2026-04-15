#!/usr/bin/env bash

# =========================
# basic run settings
# =========================
export RUN_NAME="debug_run_01"
export AGENT="both"          # rl / random / both
export EPISODES=10
export MAX_STEPS=50
export SEED=0

# =========================
# env settings
# =========================
export NUM_TAXIS=5
export GRID_SIZE=10
export MAX_ORDERS=5

# =========================
# rl hyperparameters
# =========================
export N_STEP=3
export ALPHA=0.1
export GAMMA=0.95
export EPSILON=0.2

# =========================
# run
# =========================
python  test_main.py