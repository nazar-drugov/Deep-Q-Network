# Deep-Q-Network

This repo contains an implementation of DQN for training an agent in the CartPole environment. 
The implementation uses experience replay, a target network, and epsilon-greedy exploration with linear decay.

Key components:
- QNetwork: neural network approximating Q(s, a)
- ReplayBuffer: stores (s, a, r, d, s') transitions for off-policy learning
- DQNAgent: collects experience via epsilon-greedy policy
- DQNTrainer: orchestrates training loop (replay fill, collect, train, target sync)

### Demo
https://github.com/user-attachments/assets/0b142665-05ba-4ff9-a66c-8420890f8538


https://github.com/user-attachments/assets/4e08e1f8-685f-4ca4-ba02-0a7f808f4ac6

