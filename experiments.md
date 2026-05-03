# Taxi Dispatch RL — Experiment Plan

## 1. Objective

Train an n-step SARSA agent to optimize single-taxi repositioning in a Semi-Markov Decision Process (SMDP). Evaluate whether RL learns a profitable dispatch policy, characterize the learned behavior, and quantify the effect of key hyperparameters.

## 2. Environment Configuration

| Parameter | Grid Env | Graph Env |
|---|---|---|
| Zones | 25 (5×5) | ~42 (Nansha road nodes) |
| Episode length | 96 time units | 96 time units |
| Travel time | Manhattan distance | Road network shortest path ÷ meters_per_step |
| Demand | Poisson(λ=0.5) per zone per step, uniform destination | (same) |
| Base fare | 5.0 | 5.0 |
| Rate per step | 1.0 | 1.0 |
| Cost empty | 1.0 | 1.0 |
| Cost occupied | 0.5 | 0.5 |
| meters_per_step | — (Manhattan) | 1000m |

## 3. Algorithm Configuration (Fixed)

| Parameter | Value |
|---|---|
| Algorithm | n-step SARSA (SMDP-aware) |
| α (learning rate) | 0.1 |
| γ (discount) | 0.95 |
| ε decay | multiplicative ×0.995 per episode |
| ε min | 0.01 |
| State encoder | (zone, time) → hashable tuple |
| Action space | Discrete(n_zones) — target zone to reposition to |

## 4. Experiment Matrix

### Experiment A: Grid Environment — RL vs Random Baseline

**Purpose**: Establish that RL learns a non-trivial policy on the simplified grid.

| Parameter | Train | Eval |
|---|---|---|
| Episodes | 10,000 | 200 |
| n (lookahead) | 1 | 1 |
| ε (exploration) | 0.2 → 0.01 (decay) | 0.0 (greedy) |
| Seed | 42 | 42 |
| Random baseline | — | 200 episodes (same seed) |

**Controlled variables**: env=grid, n=1, seed=42.
**Independent variable**: Agent type (RL-trained vs Random).
**Measured metrics**: All 8 metrics from Section 5.

---

### Experiment B: Graph Environment — RL vs Random Baseline

**Purpose**: Test whether RL generalizes to a real road network with irregular topology.

| Parameter | Train | Eval |
|---|---|---|
| Episodes | 10,000 | 200 |
| n (lookahead) | 1 | 1 |
| ε (exploration) | 0.2 → 0.01 (decay) | 0.0 (greedy) |
| Seed | 42 | 42 |
| Random baseline | — | 200 episodes (same seed) |
| meters_per_step | 1000 | 1000 |

**Controlled variables**: env=graph, n=1, meters_per_step=1000, seed=42.
**Independent variable**: Agent type (RL-trained vs Random).
**Measured metrics**: All 8 metrics.

---

### Experiment C: n-step Sweep (Grid)

**Purpose**: Measure the effect of lookahead depth (n) on learning speed and final policy quality.

| Run | n | Episodes | ε | Seed |
|---|---|---|---|---|
| C1 | 1 | 5,000 | 0.2 | 42 |
| C2 | 3 | 5,000 | 0.2 | 42 |
| C3 | 5 | 5,000 | 0.2 | 42 |

**Controlled variables**: env=grid, α=0.1, γ=0.95, ε=0.2, seed=42.
**Independent variable**: n (1, 3, 5).
**Eval**: Each trained model evaluated 200 episodes with ε=0.

---

### Experiment D: meters_per_step Sweep (Graph)

**Purpose**: Temporal granularity affects how many steps it takes to traverse the road network. Smaller meters_per_step = finer time resolution = more decision points.

| Run | meters_per_step | Episodes | n | Seed |
|---|---|---|---|---|
| D1 | 300 | 5,000 | 1 | 42 |
| D2 | 500 | 5,000 | 1 | 42 |
| D3 | 800 | 5,000 | 1 | 42 |
| D4 | 1000 | 5,000 | 1 | 42 |

**Controlled variables**: env=graph, n=1, seed=42.
**Independent variable**: meters_per_step.

---

## 5. Evaluation Metrics

### Core Performance (Did it learn the task?)

| Metric | Definition | Interpretation |
|---|---|---|
| `total_reward` | Σ reward per episode | Higher = better overall profit |
| `profit_per_time` | total_reward / Σ time_elapsed | Profit efficiency; fair comparison across different episode durations |
| `completed_orders` | Orders served per episode | Absolute throughput |
| `completion_rate` | completed / total_orders_generated | What fraction of demand was served |
| `empty_drive_ratio` | empty_time / (empty_time + occupied_time) | Lower = less wasteful cruising; 0.5 means half the time is spent empty |

### Behavioral (What strategy did it learn?)

| Metric | Definition | Interpretation |
|---|---|---|
| `reposition_rate` | P(action ≠ current_zone) | How often the agent actively moves to another zone |
| `stay_and_wait_rate` | P(action == current_zone) | How often it waits in place |
| `mean_trip_fare` | Average fare of completed trips | Higher may indicate preference for long-distance high-fare orders |

## 6. Output Structure per Run

```
outputs/run_<name>_<timestamp>/
├── config.json                    # All parameters
├── model.pkl                      # Final trained Q-table
├── checkpoints/
│   ├── best.pkl                   # Best episode checkpoint
│   └── checkpoint_index.json
├── trained_reward_history.pkl     # (episode, reward) pairs
├── trained_reward_curve.png       # Learning curve plot
├── trained_trajectory.pkl         # Full step-level trajectory
├── trained_episode_metrics.csv    # Per-episode metrics
├── trained_summary_metrics.json   # Aggregated metrics
├── random_reward_history.pkl
├── random_reward_curve.png
├── random_trajectory.pkl
├── random_episode_metrics.csv
├── random_summary_metrics.json
└── visuals/                       # Trajectory plots & animations
    ├── grid_compare_episode.png
    ├── grid_compare_episode.gif
    └── ...
```

## 7. Results

### Experiment A: Grid Environment — RL vs Random

| Metric | RL (n=1) | Random | Δ |
|---|---|---|---|
| **total_reward** | **+72.21** | -30.68 | **+102.89** |
| **profit_per_time** | **+0.362** | -0.154 | **+0.516** |
| **completed_orders** | **26.60** | 17.19 | **+9.41** |
| **completion_rate** | 2.22% | 2.20% | +0.02% |
| **empty_drive_ratio** | **58.7%** | 72.7% | **-14.0%** |
| **reposition_rate** | **78.8%** | 96.0% | -17.3% |
| **stay_and_wait_rate** | **21.2%** | 4.0% | +17.3% |
| **mean_trip_fare** | 8.11 | 8.20 | -0.09 |

**Key findings:**
- RL turns a loss-making random policy (-30.7) into a profitable one (+72.2).
- RL completes 55% more orders (26.6 vs 17.2).
- RL learned to **stay and wait** (21.2% vs 4.0%) rather than blindly repositioning — it recognized that some zones are statistically worth staying in.
- Empty drive ratio dropped from 72.7% to 58.7%, meaning RL wastes less fuel cruising empty.
- Learning gain (second half vs first half): +25.76 reward.

**Best episode**: #7404, reward=158.0, profit_per_time=0.79.

---

### Experiment B: Graph Environment — RL vs Random

| Metric | RL (n=1) | Random | Δ |
|---|---|---|---|
| **total_reward** | **+41.46** | -23.49 | **+64.95** |
| **profit_per_time** | **+0.208** | -0.118 | **+0.326** |
| **completed_orders** | **25.80** | 18.70 | **+7.10** |
| **completion_rate** | 0.96% | 0.95% | +0.01% |
| **empty_drive_ratio** | **64.1%** | 72.5% | **-8.4%** |
| **reposition_rate** | **92.7%** | 98.3% | -5.5% |
| **stay_and_wait_rate** | **7.3%** | 1.7% | +5.5% |
| **mean_trip_fare** | 7.80 | 7.96 | -0.16 |

**Key findings:**
- RL turns a loss-making random policy (-23.5) into a profitable one (+41.5), though the margin is smaller than on grid.
- RL completes 38% more orders (25.8 vs 18.7).
- The graph env is harder: 58 zones (vs 25 grid), irregular topology, some node pairs have no path.
- Completion rate is lower (~0.96%) because more zones = more orders generated per step = lower per-order service rate.
- RL learned to stay_and_wait more (7.3% vs 1.7%), but less than on grid (21.2%) — the road network topology makes "staying in one place" less viable.
- Learning gain (second half vs first half): +40.89 reward.

**Best episode**: #9226, reward=150.0, profit_per_time=0.75.

---

### Experiment C: n-step Sweep (Grid)

| Run | n | total_reward | profit_per_time | completed | completion_rate | empty_drive | reposition | stay_wait | trip_fare |
|---|---|---|---|---|---|---|---|---|---|
| C1 | 1 | **58.74** | **0.295** | **25.4** | 2.22% | **60.5%** | **80.8%** | **19.2%** | 8.12 |
| C2 | 3 | 39.12 | 0.196 | 24.0 | 2.21% | 63.2% | 86.7% | 13.3% | 8.08 |
| C3 | 5 | 32.59 | 0.164 | 23.3 | 2.22% | 64.0% | 87.5% | 12.5% | 8.10 |

**Best episode**: C1 (n=1) ep #3504, reward=156.5; C2 (n=3) ep #4505, reward=134.5; C3 (n=5) ep #4163, reward=121.5.

**Key finding: n=1 outperforms n=3 and n=5.** Longer lookahead does NOT help in this stochastic SMDP. With Poisson demand and random order matching, multi-step returns have high variance. Shorter n (more TD-like) learns more stably. As n increases, the agent becomes more conservative: higher reposition rate (80.8% → 87.5%) and lower stay_and_wait (19.2% → 12.5%), suggesting over-correction from noisy long-horizon returns.

---

### Experiment D: meters_per_step Sweep (Graph)

| Run | m/s | total_reward | profit_per_time | completed | completion_rate | empty_drive | reposition | stay_wait | trip_fare |
|---|---|---|---|---|---|---|---|---|---|
| D1 | 300 | **-71.28** | **-0.361** | 7.4 | 0.91% | 69.4% | 94.8% | 5.2% | 13.56 |
| D2 | 500 | -39.43 | -0.199 | 12.5 | 0.94% | 67.7% | 94.3% | 5.7% | 10.27 |
| D3 | 800 | -0.49 | -0.003 | 19.5 | 0.95% | 67.0% | 94.0% | 6.0% | 8.42 |
| D4 | 1000 | **22.51** | **0.113** | **23.7** | 0.96% | 66.6% | 94.0% | 6.0% | 7.83 |

**Key finding: Larger meters_per_step (coarser time granularity) performs better.** 
- m/s=1000 is the only profitable setting. m/s=300/500 lose money even after training.
- Smaller m/s inflates travel times (steps per km), reducing total actions per episode (17 steps at m/s=300 vs 65 at m/s=1000). Fewer actions = fewer learning opportunities.
- trip_fare increases at smaller m/s (13.56 vs 7.83) because `fare = base + rate × trip_steps`, but the higher per-trip margin cannot compensate for the drastically reduced throughput (7.4 vs 23.7 orders).
- This highlights a critical design choice: **temporal granularity must balance throughput vs per-trip economics**.

## 8. Notes

- All experiments use the same random seed (42) for reproducibility.
- Eval always uses ε=0 (greedy) to measure pure policy quality, not exploration noise.
- Random baseline uses `RandomDispatchAgent` which uniformly samples a target zone at each step.
- Graph env uses cached `.graphml` files; first run will download OSM data and may take several minutes.
- Visualizations are generated for the eval episodes, primarily for qualitative inspection.
- The env does NOT include pending-order counts in the agent's observation — the agent learns purely from (zone, time) and reward feedback.
