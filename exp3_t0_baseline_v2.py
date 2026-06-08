"""
=======================================================================
exp3_t0_baseline_v2.py  ─  T0 수치 DDPG 베이스라인 (T1 24D 동일 조건)
=======================================================================

[T1 동일 조건 수정]
  T1 제안 모델은 91D state 중 24개 인덱스만 실제 사용:
    정적: category[0~7], age[8~15], gender[60~61], pvalue_level[64~66]
    동적: prev_click[88], leave_signal[89], session_step[90]
  나머지 67D는 T1 코드에서 완전히 무시됨.

  이전 마스킹(0으로 채운 91D 입력) → 67D가 네트워크에 여전히 전달됨
  이번 수정: 24D 추출 → T1과 완전히 동등한 정보 조건
  네트워크 입력 차원: 91D → 24D (Linear(24→128))

[김홍 trainer_MDP_P_.py 대비 동일하게 맞춘 항목]
  tau=0.02, seed=0, LayerNorm=view 기반, gamma=0.7,
  hidden=128, batch=32, n_episodes=2000, memory=10000, updates/step=5

[실행]
  python exp3_t0_baseline_v2.py
=======================================================================
"""

import os, random, logging, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple

import gym
import virtualTB


# =======================================================================
# [T1 동일 조건 — 24D 추출]
# =======================================================================
T1_USED_INDICES = (
    list(range(0,  8))  +   # browsing_category (8D)
    list(range(8,  16)) +   # age_level         (8D)
    list(range(60, 62)) +   # gender            (2D)
    list(range(64, 67)) +   # pvalue_level      (3D)
    [88, 89, 90]            # 동적: prev_click, leave_signal, session_step
)   # 총 24개

T1_INPUT_DIM = len(T1_USED_INDICES)   # 24


def extract_t1_features(state_array) -> np.ndarray:
    """91D state → 24D: T1이 사용하는 인덱스만 추출."""
    return np.asarray(state_array, dtype=np.float32)[T1_USED_INDICES]


# =======================================================================
# [섹션 1] 공통 유틸리티
# =======================================================================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


Transition = namedtuple('Transition',
    ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory   = []
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


class OUNoise:
    """김홍 trainer_MDP_P_.py와 완전히 동일."""
    def __init__(self, action_dimension, scale=0.1, mu=0,
                 theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale  = scale
        self.mu     = mu
        self.theta  = theta
        self.sigma  = sigma
        self.state  = np.ones(action_dimension) * mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# =======================================================================
# [섹션 2] 김홍 ddpg.py와 동일한 LayerNorm / Actor / Critic
# =======================================================================

class LayerNorm(nn.Module):
    """
    김홍 ddpg.py의 LayerNorm과 완전히 동일한 구현.
    핵심: x.view(x.size(0), -1).mean(1) 방식 사용.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine       = affine
        self.eps          = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std  = x.view(x.size(0), -1).std(1).view(*shape)
        y    = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class Actor(nn.Module):
    """
    ★ T1 동일 조건: 입력 24D (T1_INPUT_DIM)
      Linear(24→128) + LayerNorm + ReLU
      Linear(128→128) + LayerNorm + ReLU
      Linear(128→27) + Tanh
    """
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        num_outputs = action_space.shape[0]   # 27

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1     = LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2     = LayerNorm(hidden_size)
        self.mu      = nn.Linear(hidden_size, num_outputs)
        # 김홍 ddpg.py와 동일한 초기화
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        return torch.tanh(self.mu(x))


class Critic(nn.Module):
    """
    ★ T1 동일 조건: 입력 24D (T1_INPUT_DIM)
      Linear(24→128) + LayerNorm + ReLU
      Linear(128+27→128) + LayerNorm + ReLU
      Linear(128→1)
    """
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        num_outputs = action_space.shape[0]   # 27

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1     = LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2     = LayerNorm(hidden_size)
        self.V       = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, x, a):
        x = F.relu(self.ln1(self.linear1(x)))
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.ln2(self.linear2(x)))
        return self.V(x)


# =======================================================================
# [섹션 3] DDPG 에이전트 (김홍 ddpg.py와 동일한 로직)
# =======================================================================

def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

def hard_update(target, source):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(p.data)


class NumericalDDPG:
    def __init__(self, gamma, tau, hidden_size, num_inputs,
                 action_space, device):
        self.device = device
        self.gamma  = gamma
        self.tau    = tau

        self.actor          = Actor(hidden_size, num_inputs, action_space).to(device)
        self.actor_target   = Actor(hidden_size, num_inputs, action_space).to(device)
        self.actor_optim    = Adam(self.actor.parameters(), lr=1e-4)

        self.critic         = Critic(hidden_size, num_inputs, action_space).to(device)
        self.critic_target  = Critic(hidden_size, num_inputs, action_space).to(device)
        self.critic_optim   = Adam(self.critic.parameters(), lr=1e-3)

        hard_update(self.actor_target,  self.actor)
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        self.actor.eval()
        mu = self.actor(state)
        self.actor.train()
        mu = mu.data
        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).to(self.device)
        return mu.clamp(-1, 1)

    def update_parameters(self, batch):
        state_batch      = torch.cat(list(batch.state)).to(self.device)
        action_batch     = torch.cat(list(batch.action)).to(self.device)
        reward_batch     = torch.cat(list(batch.reward)).to(self.device)
        mask_batch       = torch.cat(list(batch.mask)).to(self.device)
        next_state_batch = torch.cat(list(batch.next_state)).to(self.device)

        next_action_batch        = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch,
                                                      next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch   = mask_batch.unsqueeze(1)
        expected_q   = reward_batch + (self.gamma * mask_batch
                                       * next_state_action_values)

        self.critic_optim.zero_grad()
        current_q  = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(current_q, expected_q)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = -self.critic(state_batch,
                                   self.actor(state_batch)).mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target,  self.actor,  self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, episode, rewards, val_rewards,
                        val_ctr, vlosses, plosses):
        os.makedirs("models", exist_ok=True)
        path = f"models/exp3_checkpoint_ep{episode}.pt"
        torch.save({
            "episode":      episode,
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target":self.critic_target.state_dict(),
            "rewards":      rewards,
            "val_rewards":  val_rewards,
            "val_ctr":      val_ctr,
            "value_losses": vlosses,
            "policy_losses":plosses,
        }, path)
        logging.info(f"[EXP3 체크포인트] 저장: {path}")

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(),
                   "models/exp3_t0_actor_final")
        logging.info("[EXP3] 최종 Actor 저장: models/exp3_t0_actor_final")


# =======================================================================
# [섹션 4] 학습 루프 (김홍 trainer_MDP_P_.py와 동일한 구조)
# =======================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training_exp3_baseline.log"),
        ]
    )

    env = gym.make('VirtualTB-v0')
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device     = get_device()
    batch_size = 32

    logging.info(f"[EXP3] T0 베이스라인 시작 | device: {device}")
    logging.info("[EXP3] 김홍 MDP(P) 동일 조건: tau=0.02, seed=0, LayerNorm=view 기반")

    agent = NumericalDDPG(
        gamma       = 0.7,
        tau         = 0.02,
        hidden_size = 128,
        num_inputs  = T1_INPUT_DIM,      # ★ 24D (T1 동일 조건)
        action_space= env.action_space,
        device      = device,
    )

    memory  = ReplayMemory(10000)
    ounoise = OUNoise(env.action_space.shape[0])

    rewards, val_rewards, val_ctr = [], [], []
    value_losses, policy_losses   = [], []
    total_numsteps = 0
    updates        = 0

    for i_episode in range(2000):
        # ★ T1 동일 조건: 24개 인덱스만 추출
        state = torch.Tensor([extract_t1_features(env.reset())]).to(device)
        episode_reward = 0

        while True:
            action = agent.select_action(state, ounoise).to("cpu")
            next_state_raw, reward, done, _ = env.step(action.numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            action     = torch.Tensor(action).to(device)
            mask       = torch.Tensor([not done]).to(device)
            # ★ T1 동일 조건: next_state도 24D 추출
            next_state = torch.Tensor([extract_t1_features(next_state_raw)]).to(device)
            reward_t   = torch.Tensor([reward]).to(device)

            memory.push(state, action, mask, next_state, reward_t)
            state = next_state

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch       = Transition(*zip(*transitions))
                    vl, pl      = agent.update_parameters(batch)
                    updates    += 1
                    value_losses.append(vl)
                    policy_losses.append(pl)

            if done:
                break

        rewards.append(episode_reward)
        logging.info(
            f"[EXP3-T0] Episode {i_episode:4d} | Reward: {episode_reward:.1f}"
        )

        if i_episode > 0 and i_episode % 20 == 0:
            episode_reward = 0
            episode_step   = 0
            for _ in range(50):
                # ★ T1 동일 조건: 검증 루프도 24D 추출
                state = torch.Tensor([extract_t1_features(env.reset())]).to(device)
                while True:
                    action = agent.select_action(state).cpu()
                    next_state_raw, reward, done, _ = env.step(action.numpy()[0])
                    episode_reward += reward
                    episode_step   += 1
                    state = torch.Tensor([extract_t1_features(next_state_raw)]).to(device)
                    if done:
                        break

            avg_r = episode_reward / 50
            ctr   = episode_reward / episode_step / 10
            val_rewards.append(avg_r)
            val_ctr.append(ctr)
            vl = value_losses[-1] if value_losses else 0
            pl = policy_losses[-1] if policy_losses else 0
            logging.info(
                f"[EXP3-검증] Ep {i_episode:4d} | "
                f"AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | "
                f"VLoss: {vl:.4f} | PLoss: {pl:.4f}"
            )

        if i_episode > 0 and i_episode % 500 == 0:
            agent.save_checkpoint(
                i_episode, rewards, val_rewards,
                val_ctr, value_losses, policy_losses
            )

    os.makedirs("output", exist_ok=True)
    agent.save_model()
    np.save("output/exp3_t0_rewards",      np.array(rewards))
    np.save("output/exp3_t0_val_rewards",  np.array(val_rewards))
    np.save("output/exp3_t0_val_ctr",      np.array(val_ctr))
    np.save("output/exp3_t0_value_losses", np.array(value_losses))
    np.save("output/exp3_t0_policy_losses",np.array(policy_losses))
    env.close()
    logging.info("[EXP3] 학습 완료.")


if __name__ == "__main__":
    main()