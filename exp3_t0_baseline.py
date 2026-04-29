"""
실험 3: T0 베이스라인 — 수치 기반 DDPG (김홍 MDP-P 재현)
textify/DistilBERT 없이 91D 수치를 그대로 DDPG에 입력.
목적: LM 도입 자체의 효과를 측정하기 위한 비교 기준선.
실행: python exp3_t0_baseline.py
"""
import os, random, logging, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import namedtuple
import gym
import virtualTB
from trainer_lm_ddpg import OUNoise, get_device


Transition = namedtuple('Transition',
    ('state', 'action', 'mask', 'next_state', 'reward'))


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


# ── 김홍(2023) ddpg.py의 Actor/Critic을 그대로 재현 ──────────────
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.eps   = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Actor(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super().__init__()
        self.l1  = nn.Linear(num_inputs, hidden_size)
        self.ln1 = LayerNorm(hidden_size)
        self.l2  = nn.Linear(hidden_size, hidden_size)
        self.ln2 = LayerNorm(hidden_size)
        self.mu  = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)
    def forward(self, x):
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return torch.tanh(self.mu(x))

class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super().__init__()
        self.l1  = nn.Linear(num_inputs, hidden_size)
        self.ln1 = LayerNorm(hidden_size)
        self.l2  = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2 = LayerNorm(hidden_size)
        self.V   = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)
    def forward(self, x, a):
        x = F.relu(self.ln1(self.l1(x)))
        x = torch.cat([x, a], dim=-1)
        x = F.relu(self.ln2(self.l2(x)))
        return self.V(x)


class NumericalDDPG:
    """김홍(2023) DDPG 재현. LM 없이 91D 수치 직접 입력."""
    def __init__(self, gamma, tau, hidden_size, num_inputs, num_outputs, device):
        self.device = device
        self.gamma  = gamma
        self.tau    = tau
        self.actor         = Actor(num_inputs, hidden_size, num_outputs).to(device)
        self.actor_target  = Actor(num_inputs, hidden_size, num_outputs).to(device)
        self.critic        = Critic(num_inputs, hidden_size, num_outputs).to(device)
        self.critic_target = Critic(num_inputs, hidden_size, num_outputs).to(device)
        self.actor_optim   = Adam(self.actor.parameters(),  lr=1e-4)
        self.critic_optim  = Adam(self.critic.parameters(), lr=1e-3)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(state)
        self.actor.train()
        return mu.clamp(-1, 1)

    def update(self, batch):
        s  = torch.cat(list(batch.state)).to(self.device)
        a  = torch.cat(list(batch.action)).to(self.device)
        r  = torch.cat(list(batch.reward)).unsqueeze(1).to(self.device)
        m  = torch.cat(list(batch.mask)).unsqueeze(1).to(self.device)
        ns = torch.cat(list(batch.next_state)).to(self.device)

        with torch.no_grad():
            na  = self.actor_target(ns)
            nq  = self.critic_target(ns, na)
            tgt = r + self.gamma * m * nq

        vl = F.mse_loss(self.critic(s, a), tgt)
        self.critic_optim.zero_grad(); vl.backward(); self.critic_optim.step()

        pl = -self.critic(s, self.actor(s)).mean()
        self.actor_optim.zero_grad(); pl.backward(); self.actor_optim.step()

        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(tp.data*(1-self.tau) + p.data*self.tau)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data*(1-self.tau) + p.data*self.tau)

        return vl.item(), pl.item()

    def save_checkpoint(self, episode, rewards, val_rewards, val_ctr, vl, pl):
        os.makedirs("models", exist_ok=True)
        path = f"models/exp3_checkpoint_ep{episode}.pt"
        torch.save({
            "episode": episode, "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "rewards": rewards, "val_rewards": val_rewards,
            "val_ctr": val_ctr, "value_losses": vl, "policy_losses": pl,
        }, path)
        logging.info(f"[EXP3 체크포인트] {path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("training_exp3_baseline.log")]
    )

    device = get_device()
    env = gym.make('VirtualTB-v0')
    env.seed(2); np.random.seed(2); torch.manual_seed(2)

    num_inputs  = env.observation_space.shape[0]  # 91
    num_outputs = env.action_space.shape[0]        # 27

    agent   = NumericalDDPG(0.7, 0.003, 128, num_inputs, num_outputs, device)
    memory  = ReplayMemory(10000)
    ounoise = OUNoise(num_outputs)
    batch_size = 32; n_episodes = 2000

    logging.info(f"[EXP3] T0 베이스라인 시작 | 입력: {num_inputs}D 수치 (LM 없음)")

    rewards, val_rewards, val_ctr, vlosses, plosses = [], [], [], [], []

    for i_ep in range(n_episodes):
        state = torch.FloatTensor(np.array([env.reset()])).to(device)
        ounoise.reset()
        episode_reward = 0

        while True:
            with torch.no_grad():
                action = agent.select_action(state)
            noise = torch.FloatTensor(ounoise.noise()).to(device)
            action = (action + noise).clamp(-1, 1)

            next_state_raw, reward, done, _ = env.step(action.cpu().numpy().squeeze())
            next_state = torch.FloatTensor([next_state_raw]).to(device)

            memory.push(state, action, 
                    torch.tensor([not done]), 
                    next_state, torch.tensor([reward]))
            state = next_state
            episode_reward += reward

            if len(memory) > batch_size:
                for _ in range(5):
                    batch = Transition(*zip(*memory.sample(batch_size)))
                    vl, pl = agent.update(batch)
                    vlosses.append(vl); plosses.append(pl)
            if done:
                break

        rewards.append(episode_reward)
        logging.info(f"[EXP3-T0] Episode {i_ep:4d} | Reward: {episode_reward:.1f}")

        if i_ep > 0 and i_ep % 20 == 0:
            val_r, val_s = 0, 0
            for _ in range(50):
                s = torch.FloatTensor([env.reset()]).to(device)
                while True:
                    a = agent.select_action(s).cpu().numpy().squeeze()
                    ns, r, done, _ = env.step(a)
                    val_r += r; val_s += 1
                    s = torch.FloatTensor([ns]).to(device)
                    if done: break
            avg_r = val_r / 50
            ctr   = val_r / val_s / 10 if val_s > 0 else 0
            val_rewards.append(avg_r); val_ctr.append(ctr)
            vl = vlosses[-1] if vlosses else 0
            pl = plosses[-1] if plosses else 0
            logging.info(f"[EXP3-검증] Ep {i_ep:4d} | AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | VLoss: {vl:.4f} | PLoss: {pl:.4f}")

        if i_ep > 0 and i_ep % 500 == 0:
            agent.save_checkpoint(i_ep, rewards, val_rewards, val_ctr, vlosses, plosses)

    os.makedirs("output", exist_ok=True)
    torch.save(agent.actor.state_dict(), "models/exp3_t0_actor_final")
    np.save("output/exp3_t0_rewards",     np.array(rewards))
    np.save("output/exp3_t0_val_rewards", np.array(val_rewards))
    np.save("output/exp3_t0_val_ctr",     np.array(val_ctr))
    env.close()
    logging.info("[EXP3] 학습 완료.")

if __name__ == "__main__":
    main()
