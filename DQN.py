import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions, density):
        super(DeepQNetwork, self).__init__()

        # Shapes for the network
        self.input_dims = input_dims
        self.n_actions = n_actions

        # Density of the network
        self.fc1 = nn.Linear(self.input_dims, density)
        self.fc2 = nn.Linear(density, density)
        self.fc3 = nn.Linear(density, self.n_actions)

        # Optimizer for the Network
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

    def save_model(self, path):
        T.save(self.state_dict(), path + 'DQN')


class Agent:
    def __init__(self, input_dims, n_actions, datapath):

        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_min = 0.05
        self.eps_dec = 5e-4
        self.lr = 1e-3
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = 100000
        self.batch_size = 32
        self.mem_cntr = 0
        self.iter_cntr = 0

        self.datapath = datapath

        self.replace_target = 100
        self.PolicyNetwork = DeepQNetwork(
            self.lr, input_dims, n_actions, density=256)

        self.state_memory = np.zeros(
            (self.mem_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        self.PolicyNetwork.eval()
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float32).to(device)
            actions = self.PolicyNetwork.forward(state)
            action = T.argmax(actions).item()
            self.PolicyNetwork.train()
        else:
            action = np.random.choice(self.action_space)

        return action

    def save_models(self):
        self.PolicyNetwork.save_model(self.datapath)

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.PolicyNetwork.optimizer.zero_grad()

        # Extract Batches from the replay memory
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int)

        states = T.tensor(
            self.state_memory[batch], dtype=T.float32).to(device)
        new_states = T.tensor(
            self.new_state_memory[batch], dtype=T.float32).to(device)
        rewards = T.tensor(
            self.reward_memory[batch], dtype=T.float32).to(device)
        terminals = T.tensor(self.terminal_memory[batch]).to(device)

        # Optimize the network
        q_eval = self.PolicyNetwork.forward(states)
        q_next = self.PolicyNetwork.forward(new_states)
        q_next[terminals] = 0.0
        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.PolicyNetwork.loss(
            q_target, T.max(q_eval, dim=1)[0]).to(device)
        loss.backward()
        self.PolicyNetwork.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
