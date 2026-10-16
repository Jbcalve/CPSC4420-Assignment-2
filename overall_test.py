"""Compare alpha, gamma, and epsilon together; save the best Taxi Q-table."""

import csv
import json
import random
from itertools import product
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


ENV_NAME = "Taxi-v4"
EPISODES_PER_TRIAL = 1000
MAX_STEPS = 200
EVALUATION_EPISODES = 30
ALPHAS = [0.1, 0.5, 0.9]
GAMMAS = [0.8, 0.9, 0.99]
INITIAL_EPSILONS = [0.2, 0.5, 1.0]
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
SEED = 4420
ROOT = Path(__file__).resolve().parent


def train(env, alpha, gamma, initial_epsilon, seed):
    random.seed(seed)
    env.action_space.seed(seed)
    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float64)
    epsilon = initial_epsilon
    training_rewards = []

    for episode in range(EPISODES_PER_TRIAL):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        for _ in range(MAX_STEPS):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_best = int(np.argmax(q_table[next_state]))
            target = reward + gamma * q_table[next_state, next_best] * (not terminated)
            q_table[state, action] += alpha * (target - q_table[state, action])
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break

        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
        training_rewards.append(total_reward)

    return q_table, training_rewards


def evaluate(env, q_table):
    rewards = []
    successes = 0
    for episode in range(EVALUATION_EPISODES):
        state, _ = env.reset(seed=SEED + 100_000 + episode)
        total_reward = 0
        for _ in range(MAX_STEPS):
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                if terminated and reward > 0:
                    successes += 1
                break
        rewards.append(total_reward)
    return float(np.mean(rewards)), successes / EVALUATION_EPISODES


def plot_training_curves(histories):
    """Plot 100-episode moving averages, grouped into one panel per alpha."""
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(18, 5), sharey=True, constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    window = 100

    for ax, alpha in zip(axes, ALPHAS):
        for index, (gamma, epsilon, rewards) in enumerate(histories[alpha]):
            smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
            episodes = np.arange(window, len(rewards) + 1)
            ax.plot(episodes, smoothed, color=colors[index % len(colors)], linewidth=1.5,
                    label=f"gamma={gamma}, epsilon={epsilon}")
        ax.set_title(f"Learning rate alpha={alpha}")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("Total reward (100-episode moving average)")
    fig.suptitle("Taxi Q-learning across all hyperparameter combinations")
    fig.savefig(ROOT / "overall_test.png", dpi=160)
    plt.close(fig)


def plot_evaluation_scores(results, best_result):
    """Compare final greedy evaluation rewards for every parameter combination."""
    labels = [f"a{r['alpha']} g{r['gamma']} e{r['initial_epsilon']}" for r in results]
    scores = [r["evaluation_mean_reward"] for r in results]
    colors = ["#248232" if r is best_result else "#6688aa" for r in results]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(range(len(results)), scores, color=colors)
    ax.set_xticks(range(len(results)), labels, rotation=65, ha="right")
    ax.set_ylabel(f"Mean reward ({EVALUATION_EPISODES} greedy evaluation episodes)")
    ax.set_title("Taxi Q-learning: final evaluation by hyperparameter combination")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "overall_test_evaluation.png", dpi=160)
    plt.close(fig)


def main():
    env = gym.make(ENV_NAME)
    results = []
    histories = {alpha: [] for alpha in ALPHAS}
    best = None
    combinations = list(product(ALPHAS, GAMMAS, INITIAL_EPSILONS))

    for trial_number, (alpha, gamma, initial_epsilon) in enumerate(combinations, start=1):
        print(f"Trial {trial_number}/{len(combinations)}: alpha={alpha}, gamma={gamma}, epsilon={initial_epsilon}")
        q_table, training_rewards = train(
            env, alpha, gamma, initial_epsilon, seed=SEED + trial_number * 10_000
        )
        train_mean = float(np.mean(training_rewards[-100:]))
        histories[alpha].append((gamma, initial_epsilon, training_rewards))
        eval_mean, success_rate = evaluate(env, q_table)
        result = {
            "alpha": alpha,
            "gamma": gamma,
            "initial_epsilon": initial_epsilon,
            "training_mean_last_100": train_mean,
            "evaluation_mean_reward": eval_mean,
            "evaluation_success_rate": success_rate,
        }
        results.append(result)
        print(f"  evaluation mean reward={eval_mean:.2f}, success rate={success_rate:.0%}")

        # Primary selection metric is held-out mean reward; success rate breaks ties.
        score = (eval_mean, success_rate)
        if best is None or score > best["score"]:
            best = {"score": score, "result": result, "q_table": q_table.copy()}

    env.close()

    with (ROOT / "overall_test_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    np.save(ROOT / "taxi_q_table.npy", best["q_table"])
    with (ROOT / "best_hyperparameters.json").open("w", encoding="utf-8") as output:
        json.dump(best["result"], output, indent=2)

    plot_training_curves(histories)
    plot_evaluation_scores(results, best["result"])

    print("\nBest hyperparameters:")
    print(json.dumps(best["result"], indent=2))
    print(f"Saved best Q-table: {ROOT / 'taxi_q_table.npy'}")
    print(f"Saved training curves: {ROOT / 'overall_test.png'}")
    print(f"Saved evaluation chart and data: {ROOT / 'overall_test_evaluation.png'} and {ROOT / 'overall_test_results.csv'}")


if __name__ == "__main__":
    main()
