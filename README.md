# Deep Q-Networks (DQN)

A clean, well-documented implementation of the Deep Q-Network algorithm for reinforcement learning, targeting discrete action spaces (e.g., CartPole). The code uses experience replay, a target network, and epsilon-greedy exploration with a linear schedule.

## Structure

| File | Purpose |
|------|---------|
| `deep_q_networks.py` | Main DQN implementation: `QNetwork`, `ReplayBuffer`, `DQNAgent`, `DQNTrainer`, `DQNArgs`, training loop |
| `rl_utils.py` | Trajectory generation and visualization (`generate_and_plot_trajectory`) for CartPole rollouts |
| `gpu_env.py` | Vectorized CartPole (and legacy Pendulum) environments that run on GPU via PyTorch |
| `plotly_utils.py` | Plotting helpers: `to_numpy`, `imshow`, `line`, `scatter`, `bar`, `hist`, `cliffwalk_imshow`, `plot_cartpole_obs_and_dones` |

## Dependencies

- Python ≥ 3.10
- [gymnasium](https://gymnasium.farama.org/)
- [PyTorch](https://pytorch.org/)
- [numpy](https://numpy.org/)
- [jaxtyping](https://github.com/GoogleCloudPlatform/jaxtyping)
- [plotly](https://plotly.com/python/), [einops](https://einops.rocks/)
- [wandb](https://wandb.ai/) (optional, for logging)
- [matplotlib](https://matplotlib.org/), [Pillow](https://pillow.readthedocs.io/) (for trajectory GIFs)
- [tqdm](https://tqdm.github.io/)

Install from the `deep_q_nets` directory:

```bash
pip install -r requirements.txt
```

## Quick Start

Run from the `deep_q_nets` directory:

```bash
cd deep_q_nets
python deep_q_networks.py
```

Defaults: CartPole-v1, 500k timesteps, replay size 10k, batch size 128. Edit `DQNArgs` in `deep_q_networks.py` or pass overrides when constructing it.

## Configuration

`DQNArgs` in `deep_q_networks.py` controls all hyperparameters. Key options:

- `use_wandb`: log to Weights & Biases
- `steps_per_live_video`: periodically record and display CartPole rollouts during training (e.g. `5000`)
- `env_id`, `total_timesteps`, `buffer_size`, `batch_size`, `learning_rate`, `gamma`, `exploration_fraction`, `start_e`, `end_e`, etc.

## Logging and Videos

- With `use_wandb=True`, training metrics and (optionally) videos are sent to W&B.
- With `steps_per_live_video` set, trajectory GIFs are generated during training and saved under `videos/`.
- Final evaluation rollout is recorded after training via `_save_final_video`.

## License

See the root repository license.
