# Deep-Q-Network

This repo contains an implementation of DQN for training an agent in the CartPole environment. 
The implementation uses experience replay, a target network, and epsilon-greedy exploration with linear decay.

Key components:
- QNetwork: neural network approximating Q(s, a)
- ReplayBuffer: stores (s, a, r, d, s') transitions for off-policy learning
- DQNAgent: collects experience via epsilon-greedy policy
- DQNTrainer: orchestrates training loop (replay fill, collect, train, target sync)

### Demo
https://github.com/nazar-drugov/Deep-Q-Network/blob/main/assets/videos/cartpole_video_1.mp4

https://github.com/nazar-drugov/Deep-Q-Network/blob/main/assets/videos/cartpole_video_2.mp4
