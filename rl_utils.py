"""
Reinforcement learning utilities for the Deep Q-Networks (DQN) implementation.

This module provides trajectory generation and visualization (e.g., CartPole
rollouts rendered as GIFs) for use during training or evaluation.
"""

import time
from pathlib import Path

import torch as t
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from gpu_env import CartPole


def generate_and_plot_trajectory(
    trainer: "DQNTrainer",
    args: "DQNArgs",
    steps: int = 500,
    fps: int = 50,
):
    """
    Purpose:
        Run the trained DQN agent in a CartPole environment (GPU-friendly implementation),
        record frames, and produce an HTML animation plus a saved GIF. Used for live
        visualization during training or for final evaluation.

    Parameters:
     * trainer (DQNTrainer) : the DQN trainer whose agent to use (trainer.agent.policy_network)
     * args (DQNArgs) : config containing args.device for network forward
     * steps (int) : maximum number of environment steps to record (default: 500)
     * fps (int) : frames per second for the output animation (default: 50)

    Returns:
        IPython.display.HTML animation for Jupyter display. Also saves a GIF to
        `videos/cartpole_trajectory_<timestamp>.gif` and prints its path.
    """
    env = CartPole(env_count=1, device="cpu")
    obs, _ = env.reset()

    images = t.zeros((steps, *env.render().shape), dtype=t.uint8)

    for step_count in tqdm(range(steps), desc="Running trajectory"):
        img = env.render()
        images[step_count] = t.tensor(img, dtype=t.uint8)

        obs_tensor = t.tensor(obs, dtype=t.float32).unsqueeze(0).to(args.device)
        with t.no_grad():
            action_logits = trainer.agent.policy_network(obs_tensor)
            action = t.argmax(action_logits, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)

    env.close()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(images[0].numpy())

    def update(frame):
        im.set_array(images[frame].numpy())
        return [im]

    ani = FuncAnimation(
        fig, update, frames=range(step_count), blit=True, repeat=False, interval=1000 / fps
    )

    output_dir = Path("videos")
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / f"cartpole_trajectory_{time.strftime('%Y%m%d_%H%M%S')}.gif"
    ani.save(str(gif_path), writer="pillow", fps=fps)
    print(f"Video saved to: {gif_path.absolute()}")

    return HTML(ani.to_jshtml())
