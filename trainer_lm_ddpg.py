import gym
import virtualTB
import torch
import numpy as np
import logging
import random
from collections import namedtuple

# 혜리님이 작성한 모듈 임포트
from textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder, LMActor, LMCritic

# [참고: Sample_code/1. trainer_MDP(P).py] 데이터 구조 정의
Transition = namedtuple('Transition', ('state_91d', 'action', 'mask', 'next_state_91d', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# [참고: ddpg_cuda.py] 기반의 LM-DDPG 에이전트 래퍼 클래스
class LMDDPG:
    def __init__(self, gamma, tau, hidden_size, action_space, encoder, device):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # [참고: distilbert_encoder.py] LM 기반 네트워크 초기화
        self.actor = LMActor(hidden_size, action_space, encoder).to(device)
        self.actor_target = LMActor(hidden_size, action_space, encoder).to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_target = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # 타겟 네트워크 하드 업데이트 [참고: ddpg_cuda.py]
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, prompt, dynamic_numeric):
        self.actor.eval()
        with torch.no_grad():
            # [참고: distilbert_encoder.py] 텍스트와 수치를 동시에 입력
            mu = self.actor(prompt, dynamic_numeric)
        self.actor.train()
        return mu.clamp(-1, 1)

    def update_parameters(self, batch, group_extractor):
        # ReplayMemory에서 꺼낸 91D 상태를 다시 프롬프트와 수치로 변환 (메모리 절약형)
        prompts, dynamics = [], []
        next_prompts, next_dynamics = [], []
        
        for s, ns in zip(batch.state_91d, batch.next_state_91d):
            res = prepare_state(s, group_extractor)
            n_res = prepare_state(ns, group_extractor)
            prompts.append(res['prompt'])
            dynamics.append(res['dynamic_numeric'])
            next_prompts.append(n_res['prompt'])
            next_dynamics.append(n_res['dynamic_numeric'])

        dynamic_batch = torch.FloatTensor(dynamics).to(self.device)
        next_dynamic_batch = torch.FloatTensor(next_dynamics).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device).unsqueeze(1)
        mask_batch = torch.cat(batch.mask).to(self.device).unsqueeze(1)

        # Critic 업데이트 [참조: Wu et al.(2024)의 LM 기반 가치 평가]
        with torch.no_grad():
            next_action_batch = self.actor_target(next_prompts, next_dynamic_batch)
            next_q = self.critic_target(next_prompts, next_dynamic_batch, next_action_batch)
            expected_q = reward_batch + (self.gamma * mask_batch * next_q)

        curr_q = self.critic(prompts, dynamic_batch, action_batch)
        value_loss = torch.nn.functional.mse_loss(curr_q, expected_q)
        
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # Actor 업데이트 [참조: 텍스트 의미론에 기반한 정책 최적화]
        policy_loss = -self.critic(prompts, dynamic_batch, self.actor(prompts, dynamic_batch)).mean()
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # 소프트 업데이트 [참고: ddpg_cuda.py]
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return value_loss.item(), policy_loss.item()

def main():
    device = "cpu" if torch.cuda.is_available() else "cpu" #서버에서 쓰게 되면 "cuda" 로 변경 권장 (M1 환경에서는 "cpu"로 유지)
    env = gym.make('VirtualTB-v0')
    
    # [참고: Sample_code/2. trainer_MDP(G).py] K-Means 초기화
    group_extractor = GroupInfoExtractor(dataset_path="./virtualTB/SupervisedLearning/dataset.txt")
    group_extractor.fit() # 데이터셋에서 집단 트렌드 학습

    # [참고: distilbert_encoder.py] 인코더 및 에이전트 생성
    encoder = DistilBERTEncoder(trainable=False, device=device) # M1 환경 고려 Frozen 권장
    agent = LMDDPG(gamma=0.7, tau=0.003, hidden_size=128, action_space=env.action_space, encoder=encoder, device=device)
    
    memory = ReplayMemory(10000)
    batch_size = 32 # LM 연산 속도를 고려하여 작게 설정

    for i_episode in range(20): #목표 : 2000 
        state_91d = env.reset()
        episode_reward = 0

        while True:
            # 1. 텍스트화 [참고: textify.py]
            res = prepare_state(state_91d, group_extractor)
            prompt, dyn = res['prompt'], torch.FloatTensor(res['dynamic_numeric']).unsqueeze(0).to(device)
            
            # 2. 액션 선택
            action = agent.select_action(prompt, dyn).cpu()  # 27차원 텐서
            next_state_91d, reward, done, _ = env.step(action.numpy())  # [0]을 지워서 27차원 전체를 넘깁니다!
            
            # 3. 저장 (나중에 배치 처리를 위해 action.unsqueeze(0)으로 차원을 맞춰줍니다)
            memory.push(state_91d, action.unsqueeze(0), torch.Tensor([not done]), next_state_91d, torch.Tensor([reward]))
            state_91d = next_state_91d
            episode_reward += reward

            # 4. 학습 [참고: Sample_code/1. trainer_MDP(P).py 루프]
            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))
                v_loss, p_loss = agent.update_parameters(batch, group_extractor)

            if done: break
        
        logging.info(f"Episode {i_episode}: Reward {episode_reward}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()