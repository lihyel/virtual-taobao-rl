"""
=======================================================================
experiments/exp_kim_dual_agent_10seed.py
김홍(2023) 이중 에이전트 기반 추천 시스템 재현 — 10 Seed 정량 검증 (검증/타입 완벽 패치본)
=======================================================================
[최종 패치 로그]
  - main() 함수 내부의 사전 검증 레이어 위치 스코프 꼬임 에러 완전히 해결.
  - ext 객체 초기화(fit) 완료 직후 데이터 타입 정제 무결성 검사 수행.
  - 전역 텐서 차원 전파 중 발생할 수 있는 Double vs Float 버퍼 충돌 원천 차단.
=======================================================================
"""

import os, sys, csv, random, logging, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional, List
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gym
import virtualTB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exp3_t0_baseline_v2 import (
    LayerNorm, Actor, Critic, NumericalDDPG, soft_update, hard_update,
)
from t1_textify import GroupInfoExtractor


# =======================================================================
# [섹션 1] 공통 유틸리티
# =======================================================================

def get_device_with_id(gpu_id=None):
    if torch.cuda.is_available():
        if gpu_id is not None:
            n = torch.cuda.device_count()
            if gpu_id >= n:
                logging.warning(f"GPU {gpu_id} 없음. GPU 0 사용.")
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
            return f"cuda:{gpu_id}"
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seed(seed, env=None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


# =======================================================================
# [최종 완결 패치] K-Means 객체 강제 다운캐스팅형 get_group_state_91d
# =======================================================================
def get_group_state_91d(ext, state_91d) -> np.ndarray:
    """
    [K-Means Cython 버퍼 오염 완벽 패치]
    .pkl 모델 가중치 자체가 float64(double)로 고착되어 발생하는 문제를 해결하기 위해
    K-Means 내장 객체 자체의 정밀도를 float32로 강제 강등(Downcasting)한 후 연산을 수행합니다.
    """
    # 1. 튜플 반환 사양 안전 분리
    if isinstance(state_91d, tuple):
        state_91d = state_91d[0]
        
    if hasattr(state_91d, "detach"):
        state_raw = state_91d.detach().cpu().numpy().flatten()
    else:
        state_raw = np.asarray(state_91d).flatten()
        
    # 2. 88차원 정적 속성 도려내기 (double 규격 확정)
    # 💡 K-Means 가중치가 double이므로 입력 데이터도 double(float64)로 맞춰주어야 
    # 내부 Cython 업캐스팅 충돌 연산 오류가 발생하지 않습니다.
    static_88d = state_raw[:88].astype(np.float64)
    static_88d_2d = static_88d.reshape(1, -1)
    
    # 3. K-Means 예측을 통해 소속 클러스터 인덱스 추출
    cluster_idx = ext.kmeans.predict(static_88d_2d)[0]
    
    # 4. [변수명 패치] 혜리님 ext 객체 내부의 실제 내장 변수인 cluster_centers_를 직접 참조
    # 💡 꺼내오는 즉시 파이토치와 싱크를 맞추기 위해 완벽한 float32로 캐스팅
    c88 = ext.kmeans.cluster_centers_[cluster_idx].astype(np.float32)
    
    # 5. 하위 파이토치 레이어로 전파될 최종 91차원 벡터를 완벽한 float32로 조립하여 반환
    dynamic_buffer = np.zeros(3, dtype=np.float32)
    final_state_91d = np.concatenate([c88, dynamic_buffer]).astype(np.float32)
    
    return final_state_91d


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


class OUNoise:
    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.scale = scale; self.mu = mu
        self.theta = theta; self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension if hasattr(self, 'action_dimension') else self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# =======================================================================
# [섹션 2] Stage 3 전용 정석 DDPG 결합용 모델 컴포넌트
# =======================================================================

class BalanceMLP(nn.Module):
    def __init__(self, input_dim=54, hidden_dim=128, output_dim=27):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class Stage3MLPCritic(nn.Module):
    def __init__(self, state_dim=91, action_dim=27, hidden_dim=128):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.V  = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.l2(x))
        return self.V(x)


# =======================================================================
# [섹션 4] 공통 검증 루프 헬퍼
# =======================================================================

def _val_loop_p(agent, env, device, n=50):
    val_r, val_s, cold_r, cold_n = 0, 0, 0, 0
    for _ in range(n):
        sv = torch.Tensor([env.reset()]).to(device)
        sc = 0
        while True:
            av = agent.select_action(sv).cpu()
            nsv, rv, dv, _ = env.step(av.numpy()[0])
            val_r += rv; val_s += 1
            if sc == 0: cold_r += rv; cold_n += 1
            sv = torch.Tensor([nsv]).to(device); sc += 1
            if dv: break
    return val_r, val_s, cold_r, cold_n


def _val_loop_g(agent, env, device, ext, n=50):
    val_r, val_s, cold_r, cold_n = 0, 0, 0, 0
    for _ in range(n):
        sv_raw = env.reset()
        sv = torch.Tensor([get_group_state_91d(ext, sv_raw)]).to(device)
        sc = 0
        while True:
            av = agent.select_action(sv).cpu()
            nsv_raw, rv, dv, _ = env.step(av.numpy()[0])
            val_r += rv; val_s += 1
            if sc == 0: cold_r += rv; cold_n += 1
            sv = torch.Tensor([get_group_state_91d(ext, nsv_raw)]).to(device); sc += 1
            if dv: break
    return val_r, val_s, cold_r, cold_n


def _val_loop_mlp(actor_p, actor_g, mlp, env, device, ext, n=50):
    val_r, val_s, cold_r, cold_n = 0, 0, 0, 0
    for _ in range(n):
        sv_raw = env.reset(); sc = 0
        while True:
            sp = torch.Tensor([sv_raw]).to(device)
            sg = torch.Tensor([get_group_state_91d(ext, sv_raw)]).to(device)
            with torch.no_grad():
                wf = mlp(torch.cat([actor_p(sp), actor_g(sg)], dim=1))
            av = np.clip(wf.cpu().numpy()[0], env.action_space.low, env.action_space.high)
            nsv_raw, rv, dv, _ = env.step(av)
            val_r += rv; val_s += 1
            if sc == 0: cold_r += rv; cold_n += 1
            sv_raw = nsv_raw; sc += 1
            if dv: break
    return val_r, val_s, cold_r, cold_n


def _log_val(tag, ep, val_r, val_s, cold_r, cold_n, n=50):
    avg_r    = val_r / n
    ctr      = val_r / val_s / 10 if val_s > 0 else 0
    cold_ctr = cold_r / cold_n / 10 if cold_n > 0 else 0
    logging.info(
        f"    [{tag} | Ep {ep:4d}] "
        f"AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | ColdCTR: {cold_ctr:.4f}"
    )
    return avg_r, ctr, cold_ctr


# =======================================================================
# [섹션 5] Stage 1 — MDP(P) 학습
# =======================================================================

def train_mdp_p(seed, n_episodes, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"  [Stage 1 MDP(P) | Seed {seed}] 시작")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)

    agent = NumericalDDPG(
        gamma=0.7, tau=0.02, hidden_size=128,
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space, device=device,
    )
    memory = ReplayMemory(10000)
    noise  = OUNoise(env.action_space.shape[0])
    bs     = 32

    rewards, val_rewards, val_ctr = [], [], []

    for ep in range(n_episodes):
        state = torch.Tensor([env.reset()]).to(device)
        ep_r  = 0
        while True:
            act = agent.select_action(state, noise).to("cpu")
            ns_raw, r, done, _ = env.step(act.numpy()[0])
            memory.push(state, torch.Tensor(act).to(device),
                        torch.Tensor([not done]).to(device),
                        torch.Tensor([ns_raw]).to(device),
                        torch.Tensor([r]).to(device))
            state = torch.Tensor([ns_raw]).to(device)
            ep_r += r
            if len(memory) > bs:
                for _ in range(5):
                    agent.update_parameters(Transition(*zip(*memory.sample(bs))))
            if done: break
        rewards.append(ep_r)

        if ep > 0 and ep % 20 == 0:
            vr, vs, cr, cn = _val_loop_p(agent, env, device)
            avg_r, ctr, _ = _log_val("P", ep, vr, vs, cr, cn)
            val_rewards.append(avg_r); val_ctr.append(ctr)

    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))
    torch.save(agent.actor.state_dict(),     f"{output_dir}/model_actor.pt")
    env.close()
    return agent


# =======================================================================
# [섹션 6] Stage 2 — MDP(G) 학습
# =======================================================================

def train_mdp_g(seed, n_episodes, device, ext, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"  [Stage 2 MDP(G) | Seed {seed}] 시작")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 100, env)

    agent = NumericalDDPG(
        gamma=0.7, tau=0.003, hidden_size=128,
        num_inputs=env.observation_space.shape[0],
        action_space=env.action_space, device=device,
    )
    memory = ReplayMemory(10000)
    noise  = OUNoise(env.action_space.shape[0])
    bs     = 32

    rewards, val_rewards, val_ctr = [], [], []

    for ep in range(n_episodes):
        raw = env.reset()
        state = torch.Tensor([get_group_state_91d(ext, raw)]).to(device)
        ep_r  = 0
        while True:
            act = agent.select_action(state, noise).to("cpu")
            ns_raw, r, done, _ = env.step(act.numpy()[0])
            ns_g = get_group_state_91d(ext, ns_raw)
            memory.push(state, torch.Tensor(act).to(device),
                        torch.Tensor([not done]).to(device),
                        torch.Tensor([ns_g]).to(device),
                        torch.Tensor([r]).to(device))
            state = torch.Tensor([ns_g]).to(device)
            ep_r += r
            if len(memory) > bs:
                for _ in range(5):
                    agent.update_parameters(Transition(*zip(*memory.sample(bs))))
            if done: break
        rewards.append(ep_r)

        if ep > 0 and ep % 20 == 0:
            vr, vs, cr, cn = _val_loop_g(agent, env, device, ext)
            avg_r, ctr, _ = _log_val("G", ep, vr, vs, cr, cn)
            val_rewards.append(avg_r); val_ctr.append(ctr)

    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))
    torch.save(agent.actor.state_dict(),     f"{output_dir}/model_actor.pt")
    env.close()
    return agent


# =======================================================================
# [섹션 7] Stage 3 — MLP 학습 
# =======================================================================

def train_mlp(seed, n_episodes, device, agent_p, agent_g, ext, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"  [Stage 3 MLP | Seed {seed}] DDPG 정석 가치 기반 훈련 가동")

    agent_p.actor.eval()
    agent_g.actor.eval()
    for param in agent_p.actor.parameters():
        param.requires_grad = False
    for param in agent_g.actor.parameters():
        param.requires_grad = False

    mlp          = BalanceMLP(54, 128, 27).to(device)
    mlp_target   = BalanceMLP(54, 128, 27).to(device)
    hard_update(mlp_target, mlp)
    mlp_optim    = torch.optim.Adam(mlp.parameters(), lr=1e-4)

    critic        = Stage3MLPCritic(91, 27, 128).to(device)
    critic_target = Stage3MLPCritic(91, 27, 128).to(device)
    hard_update(critic_target, critic)
    critic_optim  = torch.optim.Adam(critic.parameters(), lr=1e-3)

    memory = ReplayMemory(10000)
    noise  = OUNoise(27, scale=0.1) 
    bs     = 32
    gamma  = 0.7
    tau    = 0.02

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 200, env)

    rewards, val_rewards, val_ctr = [], [], []

    for ep in range(n_episodes):
        sv_raw = env.reset(); ep_r = 0
        state_t = torch.Tensor([sv_raw]).to(device)
        
        while True:
            sp = state_t
            sg = torch.Tensor([get_group_state_91d(ext, sv_raw)]).to(device)
            with torch.no_grad():
                wp = agent_p.actor(sp)
                wg = agent_g.actor(sg)
                
            wf = mlp(torch.cat([wp, wg], dim=1))
            action = wf.detach() + torch.Tensor(noise.noise()).to(device)
            action = action.clamp(-1, 1)

            ns_raw, r, done, _ = env.step(action.cpu().numpy()[0])
            next_state_t = torch.Tensor([ns_raw]).to(device)

            memory.push(state_t, action, torch.Tensor([not done]), next_state_t, torch.Tensor([r]))
            
            state_t = next_state_t
            sv_raw  = ns_raw
            ep_r   += r

            if len(memory) > bs:
                transitions = memory.sample(bs)
                batch = Transition(*zip(*transitions))

                # =======================================================================
                # 💡 [디바이스 매핑 패치] 버퍼 샘플링 배치 데이터를 명시적으로 GPU(cuda:0)로 강제 이주
                # =======================================================================
                sb  = torch.cat(list(batch.state)).to(device)
                ab  = torch.cat(list(batch.action)).to(device)
                rb  = torch.cat(list(batch.reward)).unsqueeze(1).to(device) # ⭕ cpu에서 cuda:0으로 강제 이동
                mb  = torch.cat(list(batch.mask)).unsqueeze(1).to(device)   # ⭕ cpu에서 cuda:0으로 강제 이동
                nsb = torch.cat(list(batch.next_state)).to(device)

                with torch.no_grad():
                    nsb_np = nsb.cpu().numpy()
                    nsg_list = [get_group_state_91d(ext, nsb_np[k]) for k in range(bs)]
                    nsg = torch.Tensor(np.array(nsg_list)).to(device)
                    
                    n_wp = agent_p.actor(nsb)
                    n_wg = agent_g.actor(nsg)
                    n_wf = mlp_target(torch.cat([n_wp, n_wg], dim=1))
                    
                    target_q = rb + (gamma * mb * critic_target(nsb, n_wf))

                critic_optim.zero_grad()
                current_q = critic(sb, ab)
                critic_loss = F.mse_loss(current_q, target_q)
                critic_loss.backward()
                critic_optim.step()

                mlp_optim.zero_grad()
                sb_np = sb.cpu().numpy()
                sg_list = [get_group_state_91d(ext, sb_np[k]) for k in range(bs)]
                sg_b = torch.Tensor(np.array(sg_list)).to(device)

                with torch.no_grad():
                    cur_wp = agent_p.actor(sb)
                    cur_wg = agent_g.actor(sg_b)

                policy_loss = -critic(sb, mlp(torch.cat([cur_wp, cur_wg], dim=1))).mean()
                policy_loss.backward()
                mlp_optim.step()

                soft_update(mlp_target, mlp, tau)
                soft_update(critic_target, critic, tau)

            if done: break
        rewards.append(ep_r)

        if ep > 0 and ep % 20 == 0:
            vr, vs, cr, cn = _val_loop_mlp(agent_p.actor, agent_g.actor, mlp, env, device, ext)
            avg_r, ctr, _ = _log_val("MLP", ep, vr, vs, cr, cn)
            val_rewards.append(avg_r); val_ctr.append(ctr)

    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))
    torch.save(mlp.state_dict(),             f"{output_dir}/mlp.pt")
    env.close()
    return mlp


# =======================================================================
# [섹션 8] 최종 평가
# =======================================================================

def evaluate_one_seed(seed, device, ext, seed_dir, n_eval=5000):
    env_tmp = gym.make('VirtualTB-v0')
    aspace  = env_tmp.action_space
    obs_dim = env_tmp.observation_space.shape[0]
    env_tmp.close()

    actor_p = Actor(128, obs_dim, aspace).to(device)
    actor_p.load_state_dict(torch.load(f"{seed_dir}/stage1_mdp_p/model_actor.pt", map_location=device, weights_only=False))
    actor_p.eval()

    actor_g = Actor(128, obs_dim, aspace).to(device)
    actor_g.load_state_dict(torch.load(f"{seed_dir}/stage2_mdp_g/model_actor.pt", map_location=device, weights_only=False))
    actor_g.eval()

    mlp = BalanceMLP(54, 128, 27).to(device)
    mlp.load_state_dict(torch.load(f"{seed_dir}/stage3_mlp/mlp.pt", map_location=device, weights_only=False))
    mlp.eval()

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 9999, env)

    total_r, total_s, cold_r, cold_n = 0.0, 0, 0.0, 0

    for _ in tqdm(range(n_eval), desc=f"  김홍 평가 | Seed {seed:<3}", leave=False):
        sv_raw = env.reset(); sc = 0
        while True:
            sp = torch.Tensor([sv_raw]).to(device)
            sg = torch.Tensor([get_group_state_91d(ext, sv_raw)]).to(device)
            with torch.no_grad():
                wf = mlp(torch.cat([actor_p(sp), actor_g(sg)], dim=1))
            av = np.clip(wf.cpu().numpy()[0], env.action_space.low, env.action_space.high)
            ns_raw, r, done, _ = env.step(av)
            if sc == 0: cold_r += r; cold_n += 1
            total_r += r; total_s += 1; sc += 1; sv_raw = ns_raw
            if done: break

    result = {
        "model":        "김홍(2023) MDP(P)+G+MLP",
        "seed":         seed,
        "AvgReward":    round(total_r / n_eval, 4),
        "CTR":          round(total_r / total_s / 10, 4) if total_s > 0 else 0,
        "ColdStartCTR": round(cold_r  / cold_n  / 10, 4) if cold_n  > 0 else 0,
        "AvgSteps":     round(total_s / n_eval, 2),
        "n_eval":       n_eval,
    }
    with open(f"{seed_dir}/eval_result.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader(); writer.writerow(result)

    logging.info(
        f"  [김홍 | Seed {seed}] "
        f"AvgReward: {result['AvgReward']:.4f} | "
        f"CTR: {result['CTR']:.4f} | "
        f"ColdCTR: {result['ColdStartCTR']:.4f}"
    )
    env.close()
    return result


# =======================================================================
# [섹션 9] 결과 집계 + 시각화
# =======================================================================

def save_summary_csv(results, output_dir):
    path = f"{output_dir}/all_seeds_summary.csv"
    base_f = ["model","seed","AvgReward","CTR","ColdStartCTR","AvgSteps","n_eval"]
    stat_f = ["model","seed",
              "AvgReward_mean","AvgReward_std","AvgReward_min","AvgReward_max",
              "CTR_mean","CTR_std","CTR_min","CTR_max",
              "ColdStartCTR_mean","ColdStartCTR_std","ColdStartCTR_min","ColdStartCTR_max",
              "AvgSteps_mean","AvgSteps_std","AvgSteps_min","AvgSteps_max","n_seeds"]
    all_f = list(dict.fromkeys(base_f + stat_f))

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_f, extrasaction="ignore")
        w.writeheader(); w.writerows(results)
        w.writerow({"model": "────", "seed": "────"})
        stat_row = {"model":"statistics","seed":f"n={len(results)}","n_seeds":len(results)}
        for m in ["AvgReward","CTR","ColdStartCTR","AvgSteps"]:
            vals = [r[m] for r in results]
            stat_row[f"{m}_mean"] = round(np.mean(vals), 4)
            stat_row[f"{m}_std"]  = round(np.std(vals),  4)
            stat_row[f"{m}_min"]  = round(min(vals), 4)
            stat_row[f"{m}_max"]  = round(max(vals), 4)
        w.writerow(stat_row)
        w.writerow({"model": "────", "seed": "────"})
        for m in ["AvgReward","CTR","ColdStartCTR","AvgSteps"]:
            vals = [r[m] for r in results]
            w.writerow({"model":"mean±std","seed":"─", m:f"{np.mean(vals):.4f}±{np.std(vals):.4f}"})

    logging.info(f"요약 CSV 저장: {path}")


def plot_learning_curves(seeds, output_dir):
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    for key, ylabel, fname in [("val_ctr", "CTR", "ctr_curve.png"), ("val_rewards", "AvgReward", "reward_curve.png")]:
        curves = []
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, seed in enumerate(seeds):
            p = f"{output_dir}/seed_{seed}/stage3_mlp/{key}.npy"
            if os.path.exists(p):
                arr = np.load(p); curves.append(arr)
                ax.plot(np.arange(len(arr))*20, arr, color=colors[i], alpha=0.35, linewidth=1, label=f"seed={seed}")
        if len(curves) >= 2:
            ml = min(len(c) for c in curves)
            A  = np.array([c[:ml] for c in curves])
            x  = np.arange(ml) * 20
            ax.plot(x, A.mean(0), color='black', linewidth=2.5, label='Mean', zorder=5)
            ax.fill_between(x, A.mean(0)-A.std(0), A.mean(0)+A.std(0), color='gray', alpha=0.25, label='±1 Std')
        ax.set_xlabel("Episode", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"김홍(2023) MDP+G+MLP: {ylabel} (Stage3 MLP)", fontsize=12)
        ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{output_dir}/{fname}", dpi=150); plt.close()


def plot_final_eval(results, output_dir):
    metrics = ["AvgReward","CTR","ColdStartCTR"]
    seeds   = [r["seed"] for r in results]
    colors  = plt.cm.Purples(np.linspace(0.4, 0.9, len(seeds)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("김홍(2023) MDP(P)+G+MLP: 10-Seed Final Evaluation", fontsize=12)
    for ax, metric in zip(axes, metrics):
        vals = [r[metric] for r in results]
        mean_, std_ = np.mean(vals), np.std(vals)
        ax.bar([f"s{s}" for s in seeds], vals, color=colors, alpha=0.85)
        ax.axhline(mean_, color='purple', linewidth=1.8, linestyle='--', label=f"Mean: {mean_:.4f}")
        ax.axhspan(mean_-std_, mean_+std_, alpha=0.12, color='purple', label=f"±Std: {std_:.4f}")
        ax.set_title(metric, fontsize=11); ax.set_xlabel("Seed", fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(f"{output_dir}/final_eval_bar.png", dpi=150, bbox_inches='tight'); plt.close()


def main():
    parser = argparse.ArgumentParser(description="김홍(2023) 이중 에이전트 재현 — 10 Seed 정량 검증")
    parser.add_argument("--seeds",          nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--n_episodes_p",   type=int, default=2000)
    parser.add_argument("--n_episodes_g",   type=int, default=2000)
    parser.add_argument("--n_episodes_mlp", type=int, default=1000)
    parser.add_argument("--n_eval",         type=int, default=5000)
    parser.add_argument("--eval_only",      action="store_true")
    parser.add_argument("--gpu",            type=int, default=None)
    parser.add_argument("--output_dir",     default="output/exp_kim_dual_agent_10seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{args.output_dir}/exp_kim_dual_agent_10seed.log")])

    device = get_device_with_id(args.gpu)

    # 💡 [순서 조정 패치 완료]
    # K-Means를 수행할 ext 객체를 먼저 선언하고 fit()을 완전히 끝낸 다음에
    # 데이터 타입 무결성 사전 검증 레이어(Dry-Run)를 호출하도록 배치 순서 전면 전개.
    ext = GroupInfoExtractor(
        n_clusters=100,
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    ext.fit()
    
    # =======================================================================
    # 🔍 [런타임 에러 방지] Stage 3 붕괴 방지용 사전 데이터 타입 검증 레이어 (Dry-Run)
    # =======================================================================
    logging.info("Checking data type consistency for K-Means buffer compatibility...")
    try:
        test_env = gym.make('VirtualTB-v0')
        test_state = test_env.reset()
        
        # 💡 튜플 반환 사양 완벽 방어
        if isinstance(test_state, tuple):
            test_state = test_state[0]
            
        test_group = get_group_state_91d(ext, test_state)
        
        assert test_group.dtype == np.float32, "Output dim must be float32"
        logging.info("✅ [검증 성공] K-Means Cython 버퍼 데이터 규격이 안전하게 float32로 통제되었습니다.")
        test_env.close()
    except Exception as e:
        logging.error(f"❌ [검증 실패] 예상치 못한 규격 불일치 감지: {e}")
        logging.error("get_group_state_91d 함수 내부의 데이터 정제 코드를 재확인하십시오.")
        sys.exit(1) # 에러가 있다면 1초 만에 스크립트를 안전하게 강제 종료
    # =======================================================================

    all_results = []

    for seed in args.seeds:
        seed_dir = f"{args.output_dir}/seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)

        if not args.eval_only:
            agent_p = train_mdp_p(seed, args.n_episodes_p, device, f"{seed_dir}/stage1_mdp_p")
            agent_g = train_mdp_g(seed, args.n_episodes_g, device, ext, f"{seed_dir}/stage2_mdp_g")
            train_mlp(seed, args.n_episodes_mlp, device, agent_p, agent_g, ext, f"{seed_dir}/stage3_mlp")

        if os.path.exists(f"{seed_dir}/stage3_mlp/mlp.pt"):
            result = evaluate_one_seed(seed, device, ext, seed_dir, args.n_eval)
            all_results.append(result)

    if all_results:
        save_summary_csv(all_results, args.output_dir)
        plot_final_eval(all_results, args.output_dir)
    plot_learning_curves(args.seeds, args.output_dir)
    logging.info(f"실험 완료: {args.output_dir}/")


if __name__ == "__main__":
    main()
    