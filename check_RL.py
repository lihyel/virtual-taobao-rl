# Usage for Reinforcement Learning 

#Test File  
# docker container intallation and configuration check
# Offcual Github Guide 

import gym
import virtualTB

env = gym.make('VirtualTB-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)
state = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    
    if done: break
env.render()
