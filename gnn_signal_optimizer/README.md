# GNN Signal Optimizer

A production-oriented reference implementation of a graph neural network (GNN) traffic signal controller for multi-intersection SUMO simulations. The system connects to SUMO via TraCI, builds dynamic heterogeneous graphs, runs a recurrent heterogeneous graph attention network (HGAT) with Monte Carlo Dropout to choose signal phases, enforces safety constraints, and streams telemetry through a FastAPI backend.

## Architecture Overview

```
+----------------------+        +------------------+         +-------------------+
|   SUMO / TraCI       |        |   Graph Builder  |         |    HGAT Policy    |
| - junction state     |  -->   | - HeteroData     |  -->    | - MC Dropout      |
| - lane subscriptions |        | - normalization  |         | - action masking  |
+----------+-----------+        +---------+--------+         +---------+---------+
           |                               |                            |
           v                               v                            v
      Control Translator             KPI Aggregator                Inference Engine
           |                               |                            |
           v                               v                            v
        SUMO TLS                     FastAPI Server <----- WebSocket -----> Dashboard
```

The repository is organized around modular subsystems for simulation, graph processing, modeling, training, serving, and orchestration. Each module exposes typed interfaces and provides docstrings for clarity.

## Prerequisites

* Python 3.11
* [SUMO](https://www.eclipse.org/sumo/) installed and available in `PATH`
* `libsumo` optional for faster execution
* Poetry or pip for dependency management

## Installation

```bash
# clone repository
cd gnn_signal_optimizer

# install dependencies via pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# optional: install dev dependencies
pip install -r requirements-dev.txt
```

Alternatively, build the Docker image (bundles SUMO):

```bash
docker build -t gnn-signal-optimizer -f docker/Dockerfile .
```

Then run via docker-compose:

```bash
docker compose -f docker/docker-compose.yml up
```

## Example Workflow

1. **Record fixed-time trajectories for SSL pretraining:**
   ```bash
   python scripts/pretrain_ssl.py --config-path configs --config-name train_ssl mode=record
   ```
2. **Run self-supervised pretraining:**
   ```bash
   python scripts/pretrain_ssl.py --config-path configs --config-name train_ssl mode=train
   ```
3. **Fine-tune with MARL (PPO):**
   ```bash
   python scripts/train_marl.py --config-path configs --config-name train_marl
   ```
4. **Launch real-time inference loop:**
   ```bash
   python scripts/run_sumo_inference.py --config-path configs --config-name inference
   ```
5. **Start API server:**
   ```bash
   python scripts/start_server.py --config-path configs --config-name inference
   ```

## Configuration

Hierarchical YAML configurations are stored in `configs/` and loaded with Hydra. Key files:

* `env.yaml` — SUMO scenario, TLS IDs, step length, GUI toggle.
* `model.yaml` — HGAT architecture, dropout, MC passes.
* `control_policy.yaml` — safety constraints, uncertainty fallbacks.
* `train_ssl.yaml` — self-supervised training hyperparameters.
* `train_marl.yaml` — PPO hyperparameters and reward shaping.
* `inference.yaml` — model checkpoints and serving parameters.

## KPIs

The runtime computes both per-step and aggregated KPIs:

* Average travel time
* Total delay (s)
* Number of stops
* Throughput (vehicles served)
* Emissions proxy (speed variance heuristic)

KPIs are streamed via the FastAPI WebSocket endpoint and exposed via REST.

## Safety Fallback Logic

The control translator enforces minimal and maximal green durations, yellow/all-red interlocks, and uncertainty-aware fallbacks. When the MC Dropout uncertainty exceeds a configurable threshold, the system falls back to one of three policies (fixed-time, queue-based heuristic, or an external controller hook). Manual overrides are also exposed via the REST API.

## Testing

Run the automated test suite and collect coverage with:

```bash
pytest -q
```

## Documentation

Inline docstrings and type hints supplement this README to provide component-level documentation. Refer to the `gso/` package for detailed modules.
