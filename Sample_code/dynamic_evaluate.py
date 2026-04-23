import gymnasium as gym
import torch
import random
import virtualTB
from sklearn.cluster import KMeans
import sys
from ddpg_cuda import DDPG
import pickle
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
from collections import namedtuple


env = gym.make('VirtualTB-v0')
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_mdp_p(dynamic):
    mdp_p = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.observation_space.shape[0], 
        action_space = env.action_space, 
        device=device
        )
    mdp_p.load_model(
        actor_path=f"./models/ddpg_actor_virtualTB_MDP(P)-dynamic{dynamic}",
        critic_path=f"./models/ddpg_critic_virtualTB_MDP(P)-dynamic{dynamic}",
    )
    return mdp_p


def load_mdp_g(dynamic):
    mdp_g = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.observation_space.shape[0], 
        action_space = env.action_space, 
        device=device
        )
    mdp_g.load_model(
        actor_path=f"./models/ddpg_actor_virtualTB_MDP(G)-dynamic{dynamic}",
        critic_path=f"./models/ddpg_critic_virtualTB_MDP(G)-dynamic{dynamic}",
    )
    return mdp_g


def load_mlp(dynamic):
    mlp = DDPG(
        gamma = 0.7, 
        tau = 0.003, 
        hidden_size = 128, 
        num_inputs = env.action_space.shape[0] * 2, 
        action_space = env.action_space, 
        device=device
        )
    mlp.load_model(
        actor_path=f"./models/ddpg_actor_virtualTB_MLP-dynamic{dynamic}",
        critic_path=f"./models/ddpg_critic_virtualTB_MLP-dynamic{dynamic}",
    )
    return mlp


def infer_mdp_p(model_mdp_p, state):
    state = torch.Tensor([state]).to(device)
    action = model_mdp_p.select_action(state).cpu()
    return action


def eval_mdp_p(epoch, dynamic):
    model_mdp_p = load_mdp_p(dynamic)

    episode_reward = 0
    episode_step = 0
    for i in tqdm(range(epoch)):
        state = env.reset()
        while True:
            action = infer_mdp_p(model_mdp_p, state)

            next_state, reward, done, info = env.step(action.numpy()[0])
            episode_reward += reward
            episode_step += 1

            state = next_state
            if done:
                break

    logging.info("Model MDP(P): total numsteps: {}, average reward: {}, CTR: {}".format(episode_step, episode_reward / epoch, episode_reward / episode_step / 10))


def infer_mdp_g(model_mdp_g, kmeans, state):
    state = torch.Tensor([np.hstack((get_mean_state(state, kmeans), state[88:]))]).to(device)
    action = model_mdp_g.select_action(state).cpu()
    return action


def get_mean_state(state, kmeans):
    return kmeans.cluster_centers_[kmeans.predict([state[:88]])[0]]    


def eval_mdp_g(epoch, dynamic):
    model_mdp_g = load_mdp_g(dynamic)
    with open(f"./models/Kmeans-100.pkl", "rb") as f:
        kmeans = pickle.load(f)

    episode_reward = 0
    episode_step = 0
    for i in tqdm(range(epoch)):
        state = env.reset()
        while True:
            action = infer_mdp_g(model_mdp_g, kmeans, state)

            next_state, reward, done, info = env.step(action.numpy()[0])
            episode_reward += reward
            episode_step += 1

            state = next_state
            if done:
                break

    logging.info("Model MDP(G): total numsteps: {}, average reward: {}, CTR: {}".format(episode_step, episode_reward / epoch, episode_reward / episode_step / 10))


def infer_mlp(model_mdp_p, model_mdp_g, model_mlp, kmeans, state):
    state1 = infer_mdp_p(model_mdp_p, state)
    state2 = infer_mdp_g(model_mdp_g, kmeans, state)
    state = torch.cat((state1, state2), 1).to(device)
    action = model_mlp.select_action(state).cpu()
    return action


def eval_mlp(epoch, dynamic):
    model_mdp_p = load_mdp_p(dynamic)
    model_mdp_g = load_mdp_g(dynamic)
    model_mlp = load_mlp(dynamic)
    with open(f"./models/Kmeans-100.pkl", "rb") as f:
        kmeans = pickle.load(f)
    
    episode_reward = 0
    episode_step = 0
    for i in tqdm(range(epoch)):
        state = env.reset()
        while True:
            action = infer_mlp(model_mdp_p, model_mdp_g, model_mlp, kmeans, state)

            next_state, reward, done, info = env.step(action.numpy()[0])
            episode_reward += reward
            episode_step += 1

            state = next_state
            if done:
                break

    logging.info("Model MLP: total numsteps: {}, average reward: {}, CTR: {}".format(episode_step, episode_reward / epoch, episode_reward / episode_step / 10))


def eval_random(epoch):
    episode_reward = 0
    episode_step = 0
    for i in tqdm(range(epoch)):
        state = env.reset()
        while True:
            action = (np.random.rand(27) - 0.5 ) * 2

            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_step += 1

            state = next_state
            if done:
                break

    logging.info("Model Random: total numsteps: {}, average reward: {}, CTR: {}".format(episode_step, episode_reward / epoch, episode_reward / episode_step / 10))


def main(argv):
    dynamic = int(argv[0])
    eval_mdp_p(5000, dynamic)
    eval_mdp_g(5000, dynamic)
    eval_mlp(5000, dynamic)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    main(sys.argv[1:])
