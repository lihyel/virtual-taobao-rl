import gymnasium as gym
import torch
import random
import virtualTB
from ddpg_cuda import DDPG
import pickle
import numpy as np
import logging
from sklearn.cluster import KMeans
import pandas as pd
from collections import namedtuple

# 데이터 
# return
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

# ReplayMemory
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


# noise더하기
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
    """현재 state따라, standardized scale, K-means model, return the cluster centers"""
    return kmeans.cluster_centers_[kmeans.predict([state[:88]])[0]]    


def main():
    # 환경 
    env = gym.make('VirtualTB-v0')

    # seed
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    batch_size = 128

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # agent
    agent = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.observation_space.shape[0], 
        action_space = env.action_space, 
        device=device
        )

    dataset = np.array([np.array(row.split(","))[:88] for row in pd.read_csv("./data/dataset.txt", delimiter="\t", header=None)[0].values], dtype=float)
    kmeans = KMeans(n_clusters=100, random_state=42, n_jobs=-1)
    kmeans.fit(dataset)

    # memory
    memory = ReplayMemory(100000)

    # noise
    ounoise = OUNoise(env.action_space.shape[0])
    param_noise = None

    rewards = []
    total_numsteps = 0
    updates = 0
    val_rewards = []
    val_ctr = []
    value_losses = []
    policy_losses = []

    for i_episode in range(2000):
        state = env.reset()
        state = torch.Tensor([np.hstack((get_mean_state(state, kmeans), state[88:]))]).to(device)
        
        # 학습 
        episode_reward = 0
        while True:
            action = agent.select_action(state, ounoise, param_noise).to("cpu")
            next_state, reward, done, _ = env.step(action.numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            action = torch.Tensor(action).to(device)
            mask = torch.Tensor([not done]).to(device)
            next_state = torch.Tensor([np.hstack((get_mean_state(next_state, kmeans), next_state[88:]))]).to(device)
            reward = torch.Tensor([reward]).to(device)

            memory.push(state, action, mask, next_state, reward)
            state = next_state

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))

                    value_loss, policy_loss = agent.update_parameters(batch)

                    updates += 1

                    value_losses.append(value_loss)
                    policy_losses.append(policy_loss)

            if done:
                break
        rewards.append(episode_reward)

        # 테스트 
        if i_episode > 0 and i_episode % 20 == 0:
            episode_reward = 0
            episode_step = 0
            for i in range(50):
                # state = torch.Tensor([env.reset()]).to(device)
                state = env.reset()
                state = torch.Tensor([np.hstack((get_mean_state(state, kmeans), state[88:]))]).to(device)
                # state = torch.Tensor([get_mean_state(env.reset(), kmeans)]).to(device)
                while True:
                    action = agent.select_action(state).cpu()

                    next_state, reward, done, info = env.step(action.numpy()[0])
                    episode_reward += reward
                    episode_step += 1

                    # next_state = torch.Tensor([next_state])
                    # next_state = torch.Tensor([get_mean_state(next_state, kmeans)]).to(device)
                    next_state = torch.Tensor([np.hstack((get_mean_state(next_state, kmeans), next_state[88:]))]).to(device)

                    state = next_state.to(device)
                    if done:
                        break

            val_rewards.append(episode_reward / 50)
            val_ctr.append(episode_reward / episode_step / 10)
            # logging.info("Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10))
            logging.info("Episode: {}, total numsteps: {}, average reward: {}, CTR: {:.4f}, value_loss: {:.4f}, policy_loss: {:.4f}".format(
                i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10, value_loss, policy_loss
                ))
    env.close()
    agent.save_model("virtualTB", "MDP(G)_uniform")
    with open("./models/Kmeans_100.kpl", "wb") as f:
        pickle.dump(kmeans, f)
    np.save("./output/mdp(g)_value_loss_uniform", np.array(value_losses))
    np.save("./output/mdp(g)_policy_loss_uniform", np.array(policy_losses))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    main()

