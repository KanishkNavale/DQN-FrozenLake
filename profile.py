import os
import numpy as np
import json
import torch

import gym

import matplotlib.pyplot as plt

from DDQN import Agent
from train import encode_states


def predict_value(agent: Agent, state: np.ndarray) -> float:
    with torch.no_grad():
        state = torch.as_tensor(state, dtype=torch.float32, device=agent.network_one.device)
        value = torch.max(agent.network_one.forward(state)) + torch.max(agent.network_zero.forward(state))
        return value.item()


if __name__ == "__main__":

    # Init. path
    data_path = os.path.abspath('data')

    # Init. Environment and agent
    env = gym.make('FrozenLake8x8-v1')
    env.reset()

    agent = Agent(env=env, training=False)
    agent.load_models(data_path)

    with open(os.path.join(data_path, 'training_info.json')) as f:
        train_data = json.load(f)

    with open(os.path.join(data_path, 'testing_info.json')) as f:
        test_data = json.load(f)

    # Load all the data frames
    score = [data["Epidosic Summed Rewards"] for data in train_data]
    average = [data["Moving Mean of Episodic Rewards"] for data in train_data]
    test = [data["Test Score"] for data in test_data]

    # Process network data
    state_value = np.zeros((8, 8))
    k = 0
    for i in range(state_value.shape[0]):
        for j in range(state_value.shape[0]):
            state_value[i, j] = predict_value(agent, encode_states(env, k))
            k += 1
    state_value /= state_value.max()

    # Generate graphs
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))

    axes[0].plot(score, alpha=0.5, label='Episodic summation')
    axes[0].plot(average, label='Moving mean of 100 episodes')
    axes[0].grid(True)
    axes[0].set_xlabel('Training Episodes')
    axes[0].set_ylabel('Rewards')
    axes[0].legend(loc='best')
    axes[0].set_title('Training Profile')

    axes[1].boxplot(test)
    axes[1].grid(True)
    axes[1].set_xlabel('Test Run')
    axes[1].set_title('Testing Profile')

    axes[2].imshow(state_value)
    axes[2].set_xlabel('state')
    axes[2].set_title("Agent Value Estimation")
    for x in range(state_value.shape[0]):
        for y in range(state_value.shape[1]):
            axes[2].text(x, y, np.around(state_value[y, x], 2), c='white', weight='bold', ha='center', va='center')

    fig.tight_layout()
    plt.savefig(os.path.join(data_path, 'DDQN Agent Profiling.png'))
