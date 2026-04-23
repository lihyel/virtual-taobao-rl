import gymnasium as gym
import torch
import random
import virtualTB
import pickle
import sys
from ddpg_cuda import DDPG
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


def get_mean_state(state, kmeans):
    return kmeans.cluster_centers_[kmeans.predict([state[:88]])[0]] 


def main(args):
    n_clusters = 100
    _round = int(args[0])
    num = 10
    env = gym.make('VirtualTB-v0')

    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = 128

    device = "cuda" if torch.cuda.is_available() else "cpu"

    MLP = DDPG(
        gamma = 0.5, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.action_space.shape[0] * 2, 
        action_space = env.action_space, 
        device=device
        )

    with open(f"./models/kmeans-{n_clusters}.pkl", "rb") as f:
        kmeans = pickle.load(f)

    MDP_P = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = 88 + 1 + 2 * num, 
        action_space = env.action_space, 
        device=device
        )
    MDP_P.load_model(
        actor_path=f"models/ddpg_actor_virtualTB_MDP(P)-round{_round}",
        critic_path=f"models/ddpg_critic_virtualTB_MDP(P)-round{_round}",
    )

    MDP_G = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = 88 + 1 + 2 * num, 
        action_space = env.action_space, 
        device=device
        )
    MDP_G.load_model(
        actor_path=f"./models/ddpg_actor_virtualTB_MDP(G)-round{_round}",
        critic_path=f"./models/ddpg_critic_virtualTB_MDP(G)-round{_round}",
    )

    memory = ReplayMemory(100000)

    ounoise = OUNoise(env.action_space.shape[0])
    param_noise = None

    rewards = []
    total_numsteps = 0
    updates = 0
    val_rewards = []
    val_ctr = []
    
    value_losses = []
    policy_losses = []

    for i_episode in range(1000):
        state0 = env.reset()
        state1 = torch.Tensor([state0]).to(device)
        state2 = torch.Tensor([np.hstack((get_mean_state(state0, kmeans), state0[88:]))]).to(device)
        state = torch.cat((MDP_P.select_action(state1), MDP_G.select_action(state2)), 1)

        episode_reward = 0
        for _ in range(_round):
            action = MLP.select_action(state, ounoise, param_noise).to("cpu")
            next_state, reward, done, _ = env.step(action.numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            next_state1 = torch.Tensor([next_state]).to(device)
            next_state2 = torch.Tensor([np.hstack((get_mean_state(next_state, kmeans), next_state[88:]))]).to(device)
            next_state = torch.cat((MDP_P.select_action(next_state1), MDP_G.select_action(next_state2)), 1)

            action = torch.Tensor(action).to(device)
            mask = torch.Tensor([not done]).to(device)
            reward = torch.Tensor([reward]).to(device)

            memory.push(state, action, mask, next_state, reward)
            state = next_state

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))

                    value_loss, policy_loss = MLP.update_parameters(batch)
                    
                    value_losses.append(value_loss)
                    policy_losses.append(policy_loss)

                    updates += 1
            if done:
                break
        rewards.append(episode_reward)

        if i_episode % 20 == 0:
            episode_reward = 0
            episode_step = 0
            for i in range(50):
                state0 = env.reset()
                state1 = torch.Tensor([state0]).to(device)
                state2 = torch.Tensor([np.hstack((get_mean_state(state0, kmeans), state0[88:]))]).to(device)
                state = torch.cat((MDP_P.select_action(state1), MDP_G.select_action(state2)), 1)
                while True:
                    action = MLP.select_action(state).cpu()

                    next_state, reward, done, info = env.step(action.numpy()[0])
                    episode_reward += reward
                    episode_step += 1

                    next_state1 = torch.Tensor([next_state]).to(device)
                    next_state2 = torch.Tensor([np.hstack((get_mean_state(next_state, kmeans), next_state[88:]))]).to(device)
                    next_state = torch.cat((MDP_P.select_action(next_state1), MDP_G.select_action(next_state2)), 1)

                    state = next_state.to(device)
                    if done:
                        break

            val_rewards.append(episode_reward / 50)
            val_ctr.append(episode_reward / episode_step / 10)
            logging.info("Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10))

    env.close()
    MLP.save_model("virtualTB", f"MLP-round{_round}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    main(sys.argv[1:])

