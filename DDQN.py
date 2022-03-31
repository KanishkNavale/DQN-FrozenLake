import os
import numpy as np
from gym import Env

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffers.Uniform import ReplayBuffer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DoubleDeepQNetwork(torch.nn.Module):
    def __init__(self,
                 input_dimension: int,
                 action_dimension: int,
                 density: int = 1000,
                 learning_rate: float = 1e-4,
                 name: str = 'DoubleDQN') -> None:
        super(DoubleDeepQNetwork, self).__init__()

        self.name = name

        self.H1 = torch.nn.Linear(input_dimension, density)
        self.H2 = torch.nn.Linear(density, density)
        self.H3 = torch.nn.Linear(density, density)
        self.H4 = torch.nn.Linear(density, density)
        self.H5 = torch.nn.Linear(density, action_dimension)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device
        self.to(self.device)

    def forward(self, state) -> torch.Tensor:

        state = F.relu(self.H1(state))
        state = F.relu(self.H2(state))
        state = F.relu(self.H3(state))
        state = F.relu(self.H4(state))
        value = torch.tanh(self.H5(state))

        return value

    def pick_action(self, observation: torch.Tensor) -> int:
        self.eval()
        with torch.no_grad():
            state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            Q = self.forward(state)
            action = torch.argmax(Q, dim=-1)
            return action.cpu().numpy().item()

    def save_checkpoint(self, path: str = '') -> None:
        torch.save(self.state_dict(), os.path.join(path, self.name + '.pth'))

    def load_checkpoint(self, path: str = '') -> None:
        self.load_state_dict(torch.load(os.path.join(path, self.name + '.pth')))


class Agent():
    def __init__(self,
                 env: Env,
                 n_games: int = 1,
                 batch_size: int = 128,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 eps_min: float = 0.001,
                 eps_dec: float = 1e-3,
                 training: bool = True):

        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = learning_rate

        self.action_dim = env.action_space.n
        self.input_dim = env.observation_space.n
        self.batch_size = batch_size

        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.training = training

        self.memory = ReplayBuffer(self.env._max_episode_steps * n_games)

        self.network_zero = DoubleDeepQNetwork(input_dimension=self.input_dim,
                                               action_dimension=self.action_dim,
                                               learning_rate=learning_rate,
                                               name='NetworkZero')

        self.network_one = DoubleDeepQNetwork(input_dimension=self.input_dim,
                                              action_dimension=self.action_dim,
                                              learning_rate=learning_rate,
                                              name='NetworkOne')

    def epsilon_greedy_action(self, observation: torch.Tensor) -> int:
        if np.random.rand(1) > self.epsilon:
            action = self.network_zero.pick_action(observation)
        else:
            action = self.env.action_space.sample()

        return action

    def choose_action(self, observation: np.ndarray) -> int:
        if self.training:
            return self.epsilon_greedy_action(observation)
        else:
            return self.network_zero.pick_action(observation)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.memory.add(state, action, reward, next_state, done)

    def epsilon_update(self) -> None:
        if self.epsilon > self.eps_min:
            self.epsilon -= self.eps_dec

    def save_models(self, path: str) -> None:
        path = os.path.abspath(path)
        self.network_zero.save_checkpoint(path)
        self.network_one.save_checkpoint(path)

    def load_models(self, path: str) -> None:
        path = os.path.abspath(path)
        self.network_zero.load_checkpoint(path)
        self.network_one.save_checkpoint(path)

    def optimize(self):
        if self.memory.__len__() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.as_tensor(np.vstack(states), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.vstack(rewards), dtype=torch.float32, device=device)
        dones = torch.as_tensor(np.vstack(dones), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.vstack(actions), dtype=torch.int64, device=device)
        next_states = torch.as_tensor(np.vstack(next_states), dtype=torch.float32, device=device)

        self.network_zero.train()
        self.network_one.train()

        q0_values = torch.gather(self.network_zero(states), dim=1, index=actions)
        q1_values = torch.gather(self.network_one(states), dim=1, index=actions)

        with torch.no_grad():
            next_q0_values, _ = torch.max(self.network_zero(next_states), dim=-1, keepdim=True)
            next_q1_values, _ = torch.max(self.network_one(next_states), dim=-1, keepdim=True)
            next_q_values = torch.min(next_q0_values, next_q1_values)
            exprected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        network_zero_loss = F.huber_loss(exprected_q_values, q0_values)
        network_one_loss = F.huber_loss(exprected_q_values, q1_values)

        self.network_zero.optimizer.zero_grad()
        network_zero_loss.backward()
        self.network_zero.optimizer.step()

        self.network_one.optimizer.zero_grad()
        network_one_loss.backward()
        self.network_one.optimizer.step()

        self.epsilon_update()
