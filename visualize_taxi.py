import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

q_table = np.load("taxi_q_table.npy")

print("Using saved Taxi Q-learning agent")

NUM_EPISODES = 30
FPS = 4
VIDEO_PATH = "taxi_agent.mp4"
MAX_STEPS = 200

LANDMARKS = {
    0: ("R", (0, 0)),
    1: ("G", (0, 4)),
    2: ("Y", (4, 0)),
    3: ("B", (4, 3)),
}
ACTION_NAMES = ["SOUTH", "NORTH", "EAST", "WEST", "PICKUP", "DROPOFF"]

env = gym.make("Taxi-v4", render_mode="rgb_array")

frames = []
font = ImageFont.load_default()
rng = np.random.default_rng()


def capture_frame(lines, repeats=8):
    image = Image.fromarray(env.render()).convert("RGB")
    draw = ImageDraw.Draw(image)

    panel_height = 8 + 20 * len(lines)
    draw.rectangle([(0, 0), (image.width, panel_height)], fill=(25, 30, 40))

    for i, line in enumerate(lines):
        draw.text((8, 5 + i * 20), line, fill=(255, 255, 255), font=font)

    frame = np.array(image)
    frames.extend([frame] * repeats)


episode_results = []

for episode_number in range(1, NUM_EPISODES + 1):
    state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    terminated = truncated = False
    total_reward = 0

    capture_frame([
        "TAXI Q-LEARNING AGENT",
        f"EPISODE {episode_number} OF {NUM_EPISODES}",
        "New passenger and destination..."
    ], repeats=12)

    for step in range(1, MAX_STEPS + 1):
        taxi_row, taxi_col, passenger, destination = env.unwrapped.decode(state)

        q_values = q_table[state]
        action = int(np.argmax(q_values))

        destination_name = LANDMARKS[destination][0]
        if passenger == 4:
            passenger_status = "onboard"
        else:
            passenger_status = f"waiting at {LANDMARKS[passenger][0]}"

        action_name = ACTION_NAMES[action]

        capture_frame([
            f"EPISODE {episode_number} OF {NUM_EPISODES} | STEP {step}",
            f"Taxi position: row {taxi_row}, column {taxi_col}",
            f"Passenger: {passenger_status} | Destination: {destination_name}",
            f"Decision: {action_name}",
            f"Q-values: {q_values.round(2)}"
        ])

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    if terminated and reward > 0:
        outcome = "SUCCESSFUL DROPOFF"
    elif truncated or step >= MAX_STEPS:
        outcome = "STEP LIMIT REACHED"
    else:
        outcome = "EPISODE ENDED"

    episode_results.append(total_reward)

    capture_frame([
        f"EPISODE {episode_number} COMPLETE",
        f"Outcome: {outcome}",
        f"Total reward: {total_reward:+.0f}",
        f"Average reward: {np.mean(episode_results):.2f}"
    ], repeats=16)

env.close()

imageio.mimsave(
    VIDEO_PATH,
    frames,
    fps=FPS,
    macro_block_size=1
)

print(f"\nVideo saved: {VIDEO_PATH}")
print(f"Episodes recorded: {NUM_EPISODES}")
print(f"Successful drop-offs: {sum(r > 0 for r in episode_results)}")
print(f"Average reward: {np.mean(episode_results):.3f}")