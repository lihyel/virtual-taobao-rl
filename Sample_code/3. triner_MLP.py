import gymnasium as gym
import torch
import random
import virtualTB
import pickle
from ddpg_cuda import DDPG
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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


# input <- noise
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


# def get_mean_state(state, kmeans, scaler):
#     x = scaler.transform([state])
#     return kmeans.cluster_centers_[kmeans.predict(x)[0]]    
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
    MLP = DDPG(
        gamma = 0.5, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.action_space.shape[0] * 2, 
        action_space = env.action_space, 
        device=device
        )

    with open("./models/Kmeans_100.kpl", "rb") as f:
        kmeans = pickle.load(f)

    # dataset = np.array([np.array(row.split(",")) for row in pd.read_csv("./data/dataset.txt", delimiter="\t", header=None)[0].values], dtype=float)
    # kmeans = KMeans(n_clusters=100, random_state=42)
    # X = scaler.fit_transform(dataset)
    # with open("./models/Kmeans_100.kpl", "rb") as f:
    #     kmeans = pickle.load(f)
    # kmeans.fit(X)

    MDP_P = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.observation_space.shape[0], 
        action_space = env.action_space, 
        device=device
        )
    MDP_P.load_model(
        actor_path="models/ddpg_actor_virtualTB_MDP(P)_uniform",
        critic_path="models/ddpg_critic_virtualTB_MDP(P)_uniform",
    )

    MDP_G = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.observation_space.shape[0], 
        action_space = env.action_space, 
        device=device
        )
    MDP_G.load_model(
        actor_path="./models/ddpg_actor_virtualTB_MDP(G)_uniform",
        critic_path="./models/ddpg_critic_virtualTB_MDP(G)_uniform",
    )

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
        state0 = env.reset()
        state1 = torch.Tensor([state0]).to(device)
        state2 = torch.Tensor([np.hstack((get_mean_state(state0, kmeans), state0[88:]))]).to(device)
        state = torch.cat((MDP_P.select_action(state1), MDP_G.select_action(state2)), 1)

        # 학습 
        episode_reward = 0
        while True:
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

        # 테스트 
        if i_episode % 20 == 0:
            episode_reward = 0
            episode_step = 0
            for i in range(50):
                state0 = env.reset()
                state1 = torch.Tensor([state0]).to(device)
                # state2 = torch.Tensor([get_mean_state(state0, kmeans)]).to(device)
                state2 = torch.Tensor([np.hstack((get_mean_state(state0, kmeans), state0[88:]))]).to(device)
                state = torch.cat((MDP_P.select_action(state1), MDP_G.select_action(state2)), 1)
                while True:
                    action = MLP.select_action(state).cpu()

                    next_state, reward, done, info = env.step(action.numpy()[0])
                    episode_reward += reward
                    episode_step += 1

                    next_state1 = torch.Tensor([next_state]).to(device)
                    # next_state2 = torch.Tensor([get_mean_state(next_state, kmeans)]).to(device)
                    next_state2 = torch.Tensor([np.hstack((get_mean_state(next_state, kmeans), next_state[88:]))]).to(device)
                    next_state = torch.cat((MDP_P.select_action(next_state1), MDP_G.select_action(next_state2)), 1)

                    state = next_state.to(device)
                    if done:
                        break

            val_rewards.append(episode_reward / 50)
            val_ctr.append(episode_reward / episode_step / 10)
            logging.info("Episode: {}, total numsteps: {}, average reward: {}, CTR: {}".format(i_episode, episode_step, episode_reward / 50, episode_reward / episode_step / 10))

    env.close()

    MLP.save_model("virtualTB", "MLP_uniform")
    np.save("./output/mlp_value_loss_uniform", np.array(value_losses))
    np.save("./output/mlp_policy_loss_uniform", np.array(policy_losses))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    main()

