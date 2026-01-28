# Deep-Q-Network

This repo contains an implementation of DQN for training an agent in the CartPole environment. 
The implementation uses experience replay, a target network, and epsilon-greedy exploration with linear decay.

Key components:
- QNetwork: neural network approximating Q(s, a)
- ReplayBuffer: stores (s, a, r, d, s') transitions for off-policy learning
- DQNAgent: collects experience via epsilon-greedy policy
- DQNTrainer: orchestrates training loop (replay fill, collect, train, target sync)

### Demo
https://github.com/user-attachments/assets/631d04fd-f709-4276-b527-7cc809db1cfd

https://github.com/user-attachments/assets/20c4d1d8-c4d9-44d3-85d0-49cd756618e0

*** Credits ***
I built this project while independently working through the ARENA curriculum on technical AI safety. 
Many thanks to the ARENA team for creating the program and providing the .utils files used here!
