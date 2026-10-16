# CPSC 4420 Assignment 2: Q-Learning in Taxi

This project trains a tabular Q-learning agent for Gymnasium's Taxi environment. It includes a baseline training run, a combined search over learning hyperparameters, and a video that shows the saved agent choosing actions in Taxi.

## Setup

Use Python 3 and install the packages used by the scripts:

```bash
python -m pip install gymnasium numpy matplotlib imageio pillow imageio-ffmpeg
```

Run commands from the repository root. On Windows, use `py` instead of `python` if needed.

## Typical workflow: search, then make the video

Run the combined experiment first. It trains all 27 combinations, evaluates each resulting policy, and saves the best Q-table. Then run the visualizer, which loads that saved table.

```bash
python overall_test.py
python visualize_taxi.py
```

The search takes a little longer than a single training run because each combination trains for 1,000 episodes and is evaluated on 30 greedy episodes.

## Scripts and outputs

| File | Purpose | Outputs |
| --- | --- | --- |
| `baseline_model.py` | Trains one baseline agent (`alpha=0.1`, `gamma=0.99`, initial `epsilon=1.0`) and plots its smoothed training rewards. | `baseline_model.png` |
| `overall_test.py` | Compares all combinations of `alpha` (0.1, 0.5, 0.9), `gamma` (0.8, 0.9, 0.99), and initial `epsilon` (0.2, 0.5, 1.0). It chooses the best policy by mean reward over 30 greedy evaluation episodes; success rate breaks ties. | `overall_test.png`, `overall_test_evaluation.png`, `overall_test_results.csv`, `best_hyperparameters.json`, `taxi_q_table.npy` |
| `visualize_taxi.py` | Runs 30 greedy episodes using `taxi_q_table.npy` and overlays the state, action, Q-values, and outcome on the Taxi rendering. | `taxi_agent.mp4` |

`overall_test.png` plots each combination's 100-episode moving-average training reward, grouped into one panel per learning rate. `overall_test_evaluation.png` compares the final evaluation rewards. The CSV contains each combination's training and evaluation metrics, and the JSON records the selected settings.

Running `overall_test.py` again replaces its result files and `taxi_q_table.npy`. Run it before the visualizer whenever you want the video to use a newly selected model. The video script expects to be launched from the repository root so it can find the Q-table.

## Q-learning summary

The Taxi agent observes a discrete state and chooses among six actions: move south, north, east, or west; pick up the passenger; or drop off the passenger. It explores by taking random actions with a probability that decays during training; otherwise it chooses the action with the highest current Q-value. After each action, it updates the Q-value using the reward and the estimated value of the next state.

The combined search trains each parameter combination for 1,000 episodes, with up to 200 steps per episode, epsilon decay of 0.995, and minimum epsilon of 0.01. Evaluation uses the greedy policy, with no random exploration.

## Repository layout

All Python scripts and generated outputs are in the repository root:

```text
baseline_model.py                 Baseline trainer and plot
overall_test.py                   Combined hyperparameter search
visualize_taxi.py                 Saved-agent video generator
taxi_q_table.npy                  Best Q-table from the combined search
overall_test_results.csv          Metrics for all combinations
best_hyperparameters.json         Selected parameter values and metrics
overall_test.png                  Training reward curves
overall_test_evaluation.png       Evaluation reward comparison
taxi_agent.mp4                    Animated Taxi episodes
baseline_model.png                Baseline training reward plot
```
