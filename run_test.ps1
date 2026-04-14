$env:RUN_NAME="debug_run_01"
$env:AGENT="both"
$env:EPISODES="10"
$env:MAX_STEPS="50"

$env:NUM_TAXIS="5"
$env:GRID_SIZE="10"
$env:MAX_ORDERS="5"

$env:N_STEP="3"
$env:ALPHA="0.1"
$env:GAMMA="0.95"
$env:EPSILON="0.2"

python -m test_main