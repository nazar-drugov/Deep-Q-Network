"""
Deep Q-Networks (DQN) Implementation

This module implements the DQN algorithm for reinforcement learning, supporting
discrete action spaces (e.g., CartPole). The implementation uses experience replay,
a target network, and epsilon-greedy exploration with a linear schedule.

Key components:
    - QNetwork: Neural network approximating Q(s, a)
    - ReplayBuffer: Stores (s, a, r, d, s') transitions for off-policy learning
    - DQNAgent: Collects experience via epsilon-greedy policy
    - DQNTrainer: Orchestrates training loop (replay fill, collect, train, target sync)

The code is organized as follows:
    1. Imports and setup
    2. Utility functions (seeding, make_env)
    3. Q-network and replay buffer
    4. Exploration schedule and policy
    5. Agent and Trainer classes
    6. Main execution
"""

import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch as t
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn
from tqdm import tqdm, trange

warnings.filterwarnings("ignore")

# Add directory containing this script to path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plotly_utils import cliffwalk_imshow, line, plot_cartpole_obs_and_dones
from rl_utils import generate_and_plot_trajectory


# ============================================================================
# Global Configuration
# ============================================================================

section_dir = Path(__file__).resolve().parent

# Type alias for numpy arrays
Arr = np.ndarray

# Device selection: prefer MPS (Apple Silicon) > CUDA > CPU
device = t.device(
    "mps" if t.backends.mps.is_available()
    else "cuda" if t.cuda.is_available()
    else "cpu"
)


# ============================================================================
# Utility Functions
# ============================================================================


def set_global_seeds(seed: int) -> None:
    """
    Purpose:
        Set seeds for Python's random module, NumPy, and PyTorch to ensure
        experiments are reproducible across runs.

    Parameters:
     * seed (int) : the seed value to use for all random number generators

    Returns:
        None
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    if t.backends.mps.is_available():
        t.mps.manual_seed(seed)


def make_env(
    env_id: str,
    idx: int,
    run_name: str,
    video_log_freq: int | None = None,
    video_save_path: Path | None = None,
    seed: int = 1,
    **_: dict,
) -> Callable[[], gym.Env]:
    """
    Purpose:
        Create a factory function (thunk) that returns a single Gymnasium environment
        instance. The returned callable is used by `gym.vector.SyncVectorEnv` to create
        each environment in a vectorized setup. The environment is optionally wrapped
        with `RecordEpisodeStatistics` to track episode returns and lengths, and
        `RecordVideo` for periodic video logging (only for the first environment).

    Parameters:
     * env_id (str) : the Gymnasium environment ID to create
     * idx (int) : the index of this environment in the vectorized setup
     * run_name (str) : a unique name for this training run, used for video folder naming
     * video_log_freq (int | None) : optional frequency at which to record videos
                                     (only applies to the first environment, idx=0)
     * video_save_path (Path | None) : optional custom path for saving videos
     * seed (int) : base seed value for environment randomization
     * **_ (dict) : additional keyword arguments (unused, for compatibility)

    Returns:
        A callable function that, when called, returns a configured Gymnasium environment
        instance with appropriate wrappers applied.
    """

    def _thunk() -> gym.Env:
        # Create base environment with render mode for video recording
        env = gym.make(env_id, render_mode="rgb_array")

        # Track episode rewards / lengths in infos["episode"]
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Optionally save videos from the first environment
        if video_log_freq is not None and idx == 0:
            if video_save_path is None:
                video_folder = section_dir / "videos" / run_name
            else:
                video_folder = video_save_path / run_name

            video_folder.mkdir(parents=True, exist_ok=True)

            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_folder),
                episode_trigger=lambda ep: ep % video_log_freq == 0,
            )

        # Seed the environment (and its spaces) for reproducibility
        env.reset(seed=seed + idx)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + idx)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed + idx)

        return env

    return _thunk


# ============================================================================
# Q-Network and Replay Buffer
# ============================================================================


class QNetwork(nn.Module):
    """
    Purpose:
        Neural network that approximates the Q-function Q(s, a) for discrete actions.
        Used for value-based control; outputs Q-values per action for a given state.
    """

    layers: nn.Sequential

    def __init__(
        self, obs_shape: tuple[int], num_actions: int, hidden_sizes: list[int] = [120, 84]
    ):
        """
        Purpose:
            Initialize a Q-network neural network that approximates the Q-function
            for a given environment. The network consists of linear layers with ReLU
            activations between them, outputting Q-values for each possible action.

        Parameters:
         * obs_shape (tuple[int]) : shape of the observation space (must be 1D)
         * num_actions (int) : number of discrete actions in the environment
         * hidden_sizes (list[int]) : hidden layer sizes (default: [120, 84])

        Returns:
            None (initializes the QNetwork instance)
        """
        super().__init__()
        assert len(obs_shape) == 1, "Expecting a single vector of observations"
        
        # Initialize the attribute where we will store the layers
        self.layers = []

        # Create a list, storing input features for layer i at index i
        # (if we don't ReLU as a separate layer)
        in_features_list = [obs_shape[0]] + hidden_sizes

        # Create a list, storing output
        # features for layer i at index i
        out_features_list = hidden_sizes + [num_actions]

        # Populate the list of layers with layers
        num_linear_layers = len(in_features_list)
        last_lin_layer_idx = num_linear_layers - 1
        for layer_idx, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            self.layers.append(nn.Linear(in_features, out_features))
            
            # Add a ReLU after the linear layer, except for after the last linear layer
            if layer_idx < last_lin_layer_idx:
                self.layers.append(nn.ReLU())

        # Convert self.layers from a list into nn.Sequential
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Purpose:
            Perform a forward pass through the Q-network to compute Q-values for
            each possible action given the input observations.

        Parameters:
         * x (Tensor) : batch of observations with shape [batch_size, *obs_shape]

        Returns:
            Tensor of Q-values with shape [batch_size, num_actions] for each (s, a).
        """
        return self.layers(x)
    

@dataclass
class ReplayBufferSamples:
    """
    Purpose:
        Container for a minibatch of transitions sampled from the replay buffer,
        converted to PyTorch tensors for neural network training.

    Data represents (s_t, a_t, r_{t+1}, d_{t+1}, s_{t+1}). Note: d_{t+1} is **terminated**
    (out-of-bounds) not **done** (includes timeouts).
    """

    obs: Float[Tensor, " sample_size *obs_shape"]
    actions: Float[Tensor, " sample_size *action_shape"]
    rewards: Float[Tensor, " sample_size"]
    terminated: Bool[Tensor, " sample_size"]
    next_obs: Float[Tensor, " sample_size *obs_shape"]


class ReplayBuffer:
    """
    Purpose:
        Stores experience transitions (s, a, r, terminated, s') for off-policy learning.
        Supports add (append and evict oldest when full) and sample (random minibatch
        as ReplayBufferSamples).
    """

    rng: np.random.Generator
    obs: Float[Arr, " buffer_size *obs_shape"]
    actions: Float[Arr, " buffer_size *action_shape"]
    rewards: Float[Arr, " buffer_size"]
    terminated: Bool[Arr, " buffer_size"]
    next_obs: Float[Arr, " buffer_size *obs_shape"]

    def __init__(
        self,
        num_envs: int,
        obs_shape: tuple[int],
        action_shape: tuple[int],
        buffer_size: int,
        seed: int,
    ):
        """
        Purpose:
            Initialize an empty replay buffer for storing experience transitions
            (observations, actions, rewards, termination flags, and next observations).
            The buffer maintains a fixed maximum size and uses a random number generator
            for sampling.

        Parameters:
         * num_envs (int) : number of parallel environments
         * obs_shape (tuple[int]) : shape of observation vectors
         * action_shape (tuple[int]) : shape of action vectors
         * buffer_size (int) : maximum number of transitions to store
         * seed (int) : seed for RNG used in sampling

        Returns:
            None (initializes the ReplayBuffer instance)
        """
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.buffer_size = buffer_size
        self.rng = np.random.default_rng(seed)

        self.obs = np.empty((0, *self.obs_shape), dtype=np.float32)
        self.actions = np.empty((0, *self.action_shape), dtype=np.int32)
        self.rewards = np.empty(0, dtype=np.float32)
        self.terminated = np.empty(0, dtype=bool)
        self.next_obs = np.empty((0, *self.obs_shape), dtype=np.float32)

    def add(
        self,
        obs: Float[Arr, " num_envs *obs_shape"],
        actions: Int[Arr, " num_envs *action_shape"],
        rewards: Float[Arr, " num_envs"],
        terminated: Bool[Arr, " num_envs"],
        next_obs: Float[Arr, " num_envs *obs_shape"],
    ) -> None:
        """
        Purpose:
            Add a batch of experience transitions to the replay buffer. If the buffer
            exceeds its maximum size, the oldest transitions are removed.

        Parameters:
         * obs (Float[Arr, " num_envs *obs_shape"]) : current observations per environment
         * actions (Int[Arr, " num_envs *action_shape"]) : actions taken per environment
         * rewards (Float[Arr, " num_envs"]) : rewards received per environment
         * terminated (Bool[Arr, " num_envs"]) : whether each environment terminated
         * next_obs (Float[Arr, " num_envs *obs_shape"]) : next observations after actions

        Returns:
            None
        """
        # Check shapes & datatypes
        for data, expected_shape in zip(
            [obs, actions, rewards, terminated, next_obs],
            [self.obs_shape, self.action_shape, (), (), self.obs_shape],
        ):
            assert isinstance(data, np.ndarray)
            assert data.shape == (self.num_envs, *expected_shape)

        # Add data to buffer, slicing off the old elements
        self.obs = np.concatenate((self.obs, obs))[-self.buffer_size :]
        self.actions = np.concatenate((self.actions, actions))[-self.buffer_size :]
        self.rewards = np.concatenate((self.rewards, rewards))[-self.buffer_size :]
        self.terminated = np.concatenate((self.terminated, terminated))[-self.buffer_size :]
        self.next_obs = np.concatenate((self.next_obs, next_obs))[-self.buffer_size :]

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        """
        Purpose:
            Randomly sample a batch of transitions from the replay buffer with replacement,
            converting them to PyTorch tensors on the specified device for use in neural
            network training.

        Parameters:
         * sample_size (int) : number of transitions to sample
         * device (t.device) : PyTorch device (CPU, CUDA, or MPS) for tensors

        Returns:
            ReplayBufferSamples containing sampled transitions as PyTorch tensors.
        """
        num_stored = len(self.obs)
        assert num_stored >= sample_size, (
            f"Cannot sample {sample_size} transitions: buffer has only {num_stored}."
        )
        indices = self.rng.integers(0, num_stored, size=sample_size)

        return ReplayBufferSamples(
            obs=t.tensor(self.obs[indices], dtype=t.float32, device=device),
            actions=t.tensor(self.actions[indices], device=device),
            rewards=t.tensor(self.rewards[indices], dtype=t.float32, device=device),
            terminated=t.tensor(self.terminated[indices], device=device),
            next_obs=t.tensor(self.next_obs[indices], dtype=t.float32, device=device),
        )


# ============================================================================
# Exploration Schedule and Policy
# ============================================================================


def linear_schedule(
    current_step: int,
    start_e: float,
    end_e: float,
    exploration_fraction: float,
    total_timesteps: int,
) -> float:
    """
    Purpose:
        Compute the epsilon value for epsilon-greedy exploration using a linear schedule.
        Epsilon starts at start_e and decreases linearly to end_e over the first
        exploration_fraction fraction of total timesteps, then remains constant at end_e
        for the remainder of training.

    Parameters:
     * current_step (int) : current training step number
     * start_e (float) : initial epsilon at step 0
     * end_e (float) : final epsilon after exploration phase
     * exploration_fraction (float) : fraction of total timesteps in exploration
     * total_timesteps (int) : total training timesteps

    Returns:
        Epsilon (float) for the current step, between end_e and start_e.
    """
    # Compute number of the step when we should end exploration
    exploration_end_step = exploration_fraction * total_timesteps

    # Compute the slope k and the intercept b
    b = start_e
    k = (end_e - start_e) / exploration_end_step

    # Case 1
    if current_step <= exploration_end_step:
        return k * current_step + b
    else:
        return end_e
    

def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv,
    q_network: QNetwork,
    rng: np.random.Generator,
    obs: Float[Arr, " num_envs *obs_shape"],
    epsilon: float,
) -> Int[Arr, " num_envs *action_shape"]:
    """
    Purpose:
        Select actions for each environment using an epsilon-greedy policy. With
        probability epsilon, a random action is selected for exploration. Otherwise,
        the action with the highest Q-value according to the Q-network is selected
        for exploitation.

    Parameters:
     * envs (gym.vector.SyncVectorEnv) : the vectorized environment containing action space information
     * q_network (QNetwork) : the neural network used to approximate Q-values
     * rng (np.random.Generator) : random number generator for selecting random actions
     * obs (Float[Arr, " num_envs *obs_shape"]) : current observations for each environment
     * epsilon (float) : probability of taking a random action (between 0 and 1)

    Returns:
        Array of actions (Int[Arr, " num_envs *action_shape"]), one per environment.
    """
    num_envs = envs.num_envs
    num_actions = envs.single_action_space.n

    # Convert `obs` into a tensor so we can feed it into our model
    obs = t.from_numpy(obs).to(device)

    # Determine if we should take a random or greedy action
    behavior = "random" if rng.random() <= epsilon else "greedy"

    if behavior == "random":
        actions = rng.integers(0, num_actions, size=(num_envs,))
    else:
        actions = q_network(obs).argmax(dim=-1)
        actions = actions.detach().cpu().numpy()

    return actions


# ============================================================================
# Configuration and Agent / Trainer
# ============================================================================


@dataclass
class DQNArgs:
    """
    Purpose:
        Configuration dataclass containing all hyperparameters for DQN training.
        Computes derived values (total_training_steps, video_save_path, device) in
        __post_init__.
    """

    # === Basic Configuration ===
    seed: int = 1  # Random seed for reproducibility
    env_id: str = "CartPole-v1"  # Gymnasium environment ID
    num_envs: int = 1  # Number of parallel environments

    # === Logging Configuration ===
    use_wandb: bool = False  # Whether to log to Weights & Biases
    wandb_project_name: str = "DQNCartPole"  # W&B project name
    wandb_entity: str | None = None  # W&B entity/username (None = default)
    video_log_freq: int | None = 50  # Video log frequency (None = disabled)
    steps_per_live_video: int | None = None  # Log live videos every N steps (None = disabled)

    # === Replay Buffer and Training Phases ===
    total_timesteps: int = 500_000  # Total environment steps
    steps_per_train: int = 10  # Env steps between each training step
    trains_per_target_update: int = 100  # Training steps between target network updates
    buffer_size: int = 10_000  # Maximum replay buffer size

    # === Optimization Hyperparameters ===
    batch_size: int = 128  # Minibatch size for TD updates
    learning_rate: float = 2.5e-4  # Adam learning rate

    # === RL Hyperparameters ===
    gamma: float = 0.99  # Discount factor for future rewards
    exploration_fraction: float = 0.2  # Fraction of timesteps for linear epsilon decay
    start_e: float = 1.0  # Initial epsilon (exploration)
    end_e: float = 0.1  # Final epsilon after decay

    def __post_init__(self):
        """
        Purpose:
            Validate configuration and compute derived values (total_training_steps,
            video_save_path, device) after DQNArgs initialization.
        """
        assert self.total_timesteps - self.buffer_size >= self.steps_per_train
        self.total_training_steps = (
            self.total_timesteps - self.buffer_size
        ) // self.steps_per_train
        self.video_save_path = section_dir / "videos"
        self.device = device


class DQNAgent:
    """
    Purpose:
        Agent that interacts with vectorized environments, collects experience via
        epsilon-greedy policy, and stores transitions in a replay buffer.
    """

    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        buffer: ReplayBuffer,
        q_network: QNetwork,
        start_e: float,
        end_e: float,
        exploration_fraction: float,
        total_timesteps: int,
        rng: np.random.Generator,
    ):
        """
        Purpose:
            Initialize a DQN agent that interacts with environments, collects experience,
            and stores transitions in a replay buffer. The agent uses an epsilon-greedy
            policy with a linear exploration schedule.

        Parameters:
         * envs (gym.vector.SyncVectorEnv) : vectorized environments to interact with
         * buffer (ReplayBuffer) : replay buffer for experience transitions
         * q_network (QNetwork) : Q-network for action selection
         * start_e (float) : initial epsilon for exploration
         * end_e (float) : final epsilon after exploration phase
         * exploration_fraction (float) : fraction of timesteps in exploration
         * total_timesteps (int) : total training timesteps
         * rng (np.random.Generator) : RNG for action selection

        Returns:
            None (initializes the DQNAgent instance)
        """
        self.envs = envs
        self.buffer = buffer
        self.q_network = q_network
        self.start_e = start_e
        self.end_e = end_e
        self.exploration_fraction = exploration_fraction
        self.total_timesteps = total_timesteps
        self.rng = rng

        self.step = 0  # Tracking number of steps taken (across all environments)
        self.obs, _ = self.envs.reset()  # Need a starting observation
        self.epsilon = start_e  # Starting value (will be updated in `get_actions`)

    @property
    def policy_network(self):
        """
        Compatibility property for functions expecting a policy_network.
        Returns the Q-network, which can be used to get actions via argmax.
        """
        return self.q_network

    def play_step(self) -> dict:
        """
        Purpose:
            Execute a single interaction step between the agent and all environments.
            The agent selects actions using its epsilon-greedy policy, steps the
            environments, and stores the resulting transitions in the replay buffer.

        Parameters:
            None (uses self attributes)

        Returns:
            Dict from env step (infos), including episode stats if any terminated.
        """
        # Get actions to perform in response to observations
        self.obs = np.array(self.obs, dtype=np.float32)
        actions = self.get_actions(self.obs)

        # Step our environment with the actions
        next_obs, rewards, terminated, truncated, infos = self.envs.step(actions)

        # Make sure that next_obs stores true next observations, not reset observations
        true_next_obs = next_obs.copy()
        for idx in range(self.envs.num_envs):
            if (terminated | truncated)[idx]:
                true_next_obs[idx] = infos["final_observation"][idx]

        # Add the new observations to the buffer
        self.buffer.add(
                        self.obs,
                        actions,
                        rewards,
                        terminated,
                        true_next_obs
                        )
        self.obs = next_obs

        self.step += self.envs.num_envs
        return infos

    def get_actions(self, obs: np.ndarray) -> np.ndarray:
        """
        Purpose:
            Sample actions for the given observations using an epsilon-greedy policy.
            The epsilon value is updated according to a linear schedule based on the
            current training step, then actions are selected using epsilon_greedy_policy.

        Parameters:
         * obs (np.ndarray) : current observations per environment

        Returns:
            Array of actions (np.ndarray), one per environment.
        """
        # Set epsilon according to the linear schedule
        self.epsilon = linear_schedule(
                                       self.step,
                                       self.start_e,
                                       self.end_e,
                                       self.exploration_fraction,
                                       self.total_timesteps
                                      )

        # Sample actions according to epsilon greedy policy
        actions = epsilon_greedy_policy(
                                        self.envs,
                                        self.q_network,
                                        self.rng,
                                        obs,
                                        self.epsilon
                                        )

        return actions


def get_episode_data_from_infos(infos: dict) -> dict[str, int | float] | None:
    """
    Purpose:
        Extract episode statistics (length, reward, duration) from the first terminated
        environment in the provided info dictionary. This is used for logging training
        progress.

    Parameters:
     * infos (dict) : environment step info, including "final_info" with episode
                     data for terminated environments

    Returns:
        Dict with episode_length, episode_reward, episode_duration if any env
        terminated; otherwise None.
    """
    for final_info in infos.get("final_info", []):
        if final_info is not None and "episode" in final_info:
            return {
                "episode_length": final_info["episode"]["l"].item(),
                "episode_reward": final_info["episode"]["r"].item(),
                "episode_duration": final_info["episode"]["t"].item(),
            }
    return None


class DQNTrainer:
    def __init__(self, args: DQNArgs):
        """
        Purpose:
            Initialize a DQN trainer that sets up all components needed for training
            a Deep Q-Network agent. This includes setting random seeds, creating
            environments, initializing networks and optimizer, and creating the agent.

        Parameters:
         * args (DQNArgs) : hyperparameters and configuration for training

        Returns:
            None (initializes the DQNTrainer instance)
        """
        set_global_seeds(args.seed)
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.run_name = f"{args.env_id}__{args.wandb_project_name}__seed{args.seed}__{time.strftime('%Y%m%d-%H%M%S')}"
        # Create environments without video logging during training
        env_args = {**args.__dict__, "video_log_freq": None}
        self.envs = gym.vector.SyncVectorEnv(
            [
                make_env(idx=idx, run_name=self.run_name, **env_args)
                for idx in range(args.num_envs)
            ]
        )

        # Define some basic variables from our environment (note, we assume a single discrete action space)
        num_envs = self.envs.num_envs
        action_shape = self.envs.single_action_space.shape
        num_actions = self.envs.single_action_space.n
        obs_shape = self.envs.single_observation_space.shape
        assert action_shape == ()

        # Create our replay buffer
        self.buffer = ReplayBuffer(num_envs, obs_shape, action_shape, args.buffer_size, args.seed)

        # Create our networks & optimizer (target network should be initialized with a copy of the Q-network's weights)
        self.q_network = QNetwork(obs_shape, num_actions).to(device)
        self.target_network = QNetwork(obs_shape, num_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = t.optim.AdamW(self.q_network.parameters(), lr=args.learning_rate)

        # Create our agent
        self.agent = DQNAgent(
            self.envs,
            self.buffer,
            self.q_network,
            args.start_e,
            args.end_e,
            args.exploration_fraction,
            args.total_timesteps,
            self.rng,
        )

    def add_to_replay_buffer(self, n: int, verbose: bool = False):
        """
        Purpose:
            Execute n interaction steps with the agent, collecting experience and
            storing transitions in the replay buffer. Optionally displays a progress
            bar if verbose is True. Returns episode statistics from the most recently
            terminated episode, if any.

        Parameters:
         * n (int) : number of env steps to take with the agent
         * verbose (bool) : if True, show progress bar during execution

        Returns:
            Dict with episode_length, episode_reward, episode_duration from last
            terminated episode, or None if none terminated.
        """
        infos_to_log = []

        # Take n steps with the agent (each step adds experiences to replay buffer)
        iterator = trange(n) if verbose else range(n)
        for _ in iterator:
            infos_to_log.append(self.agent.play_step())

        # Find a dict with info for the last terminated episode, if there is any
        for i in range(n-1, -1, -1):
            infos = infos_to_log[i]
            last_terminated_data = get_episode_data_from_infos(infos)

            if last_terminated_data is not None:
                return last_terminated_data  

        return None
        
    def prepopulate_replay_buffer(self):
        """
        Purpose:
            Fill the replay buffer with initial experience before training begins.
            This ensures that the buffer has enough diverse transitions for effective
            batch sampling during the first training steps.

        Parameters:
            None (uses self.args)

        Returns:
            None
        """
        n_steps_to_fill_buffer = self.args.buffer_size // self.args.num_envs
        self.add_to_replay_buffer(n_steps_to_fill_buffer, verbose=True)

    def training_step(self, step: int) -> dict[str, float]:
        """
        Purpose:
            Perform a single training step by sampling a batch from the replay buffer,
            computing the temporal difference (TD) loss, and updating the Q-network
            weights via backpropagation. Periodically updates the target network by
            copying weights from the Q-network.

        Parameters:
         * step (int) : current training step (used for target network sync)

        Returns:
            Dict with td_loss, q_values, epsilon.
        """
        # Sample a minibatch from the replay buffer
        samples = self.buffer.sample(self.args.batch_size, device)
        obs, actions, rewards, terminated, next_obs =\
            samples.obs, samples.actions, samples.rewards, samples.terminated, samples.next_obs

        # Get the predicted Q-values
        # (a tensor [Q(s, a_1), ..., Q(s, a_n) for env in the batch])
        predicted_Q_vals = self.q_network(obs)
        predicted_Q_vals = predicted_Q_vals.gather(1, actions.unsqueeze(1))

        # Get the max target Q-values from the target network
        with t.inference_mode():
            max_target_Q_vals = self.target_network(next_obs)
            max_target_Q_vals = t.max(input=max_target_Q_vals, dim=-1, keepdim=True).values

        # Compute TD loss
        predicted_Q_vals = predicted_Q_vals.squeeze(1)
        max_target_Q_vals = max_target_Q_vals.squeeze(1)
        loss = t.mean((rewards + (1-terminated.float())*self.args.gamma*max_target_Q_vals - predicted_Q_vals)**2)

        # Perform backpropagation and take an optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log data into wandb (only when videos are being saved/displayed)
        if (
            self.args.use_wandb
            and self.args.steps_per_live_video is not None
            and step % self.args.steps_per_live_video == 0
        ):
            wandb.log({
                "td_loss": loss.item(),
                "q_values": predicted_Q_vals.mean().item(),
                "epsilon": self.agent.epsilon,
            }, step=step)

        # Sync target network every trains_per_target_update training steps
        if step > 0 and step % self.args.trains_per_target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Return metrics for display
        return {
            "td_loss": loss.item(),
            "q_values": predicted_Q_vals.mean().item(),
            "epsilon": self.agent.epsilon,
        }

    def _save_final_video(self) -> None:
        """
        Purpose:
            Create a single environment with video recording enabled and run one episode
            with the trained agent to save the final video.

        Parameters:
            None (uses self attributes)

        Returns:
            None
        """
        # Create a single env with video recording (save every episode)
        video_env_args = {**self.args.__dict__, "video_log_freq": 1, "num_envs": 1}
        video_env_fn = make_env(idx=0, run_name=self.run_name, **video_env_args)
        video_env = video_env_fn()
        
        # Run one episode with the trained agent
        obs, _ = video_env.reset()
        done = False
        while not done:
            # Use greedy policy (epsilon=0) for final video
            obs_tensor = t.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0).to(device)
            with t.no_grad():
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax(dim=-1).item()
            obs, reward, terminated, truncated, _ = video_env.step(action)
            done = terminated or truncated
        
        video_env.close()

    def train(self) -> None:
        """
        Purpose:
            Execute the main training loop for the DQN agent. This method orchestrates
            the entire training process: initializes logging, pre-populates the replay
            buffer, then iteratively collects experience, performs training steps, and
            logs progress. Optionally generates live videos of the agent's behavior
            during training.

        Parameters:
            None (uses self.args and instance attributes)

        Returns:
            None
        """
        # Initialize Weights & Biases (if requested)
        if self.args.use_wandb:
            wandb.init(
                project=self.args.wandb_project_name,
                entity=self.args.wandb_entity,
                name=self.run_name,
                monitor_gym=self.args.video_log_freq is not None,
            )
            wandb.watch(self.q_network, log="all", log_freq=50)

        # Fill the replay buffer before training so that each batch has enough diversity
        self.prepopulate_replay_buffer()

        # Create a progress bar over the number of training steps
        pbar = tqdm(range(self.args.total_training_steps))

        # Track when we last updated the progress bar with new episode statistics
        last_logged_time = time.time()
        
        # Track training metrics for display
        metrics = {"td_loss": 0.0, "q_values": 0.0, "epsilon": 0.0}

        # Main training loop: collect experience, then train on it
        for step in pbar:
            # Collect a batch of new transitions and (optionally) get info from a finished episode
            data = self.add_to_replay_buffer(self.args.steps_per_train)

            # Take a single gradient step using a batch from the replay buffer
            step_metrics = self.training_step(step)
            metrics.update(step_metrics)

            # Update the progress bar with latest episode stats and training metrics
            display_data = {**metrics}
            if data is not None:
                display_data.update(data)
            
            if time.time() - last_logged_time > 0.5:
                last_logged_time = time.time()
                pbar.set_postfix(**display_data)

            # Optionally generate and display a live video of the agent's behavior
            if (
                self.args.steps_per_live_video is not None
                and step % self.args.steps_per_live_video == 0
            ):
                from IPython.display import display

                html_animation = generate_and_plot_trajectory(self, self.args)
                display(html_animation)

        # Close training environments
        self.envs.close()
        
        # Save final video with trained agent
        self._save_final_video()
        
        if self.args.use_wandb:
            wandb.finish()


# ============================================================================
# Main Execution
# ============================================================================


if __name__ == "__main__":
    args = DQNArgs(use_wandb=True, steps_per_live_video=5_000)
    trainer = DQNTrainer(args)
    trainer.train()