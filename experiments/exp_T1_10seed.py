"""
=======================================================================
experiments/exp_kim_dual_agent_10seed.py
김홍(2023) 이중 에이전트 기반 추천 시스템 재현 — 10 Seed 정량 검증
=======================================================================

[참고 문헌]
  김홍 (2023). 이중 에이전트 기반 강화학습 추천 시스템.
  (석사 학위 논문, 이화여자대학교)

[아키텍처 — 선배 논문 설계 명세 기반 재구현]

  MDP(P): 91D state → DDPG(hidden=128) → 27D  [tau=0.02, gamma=0.7]
  MDP(G): K-Means centroid 91D → DDPG(hidden=128) → 27D  [tau=0.003]
  MLP:    concat[27D+27D]=54D → Linear(128) → Linear(27)

[3단계 학습 순서]
  Stage 1. MDP(P): n_episodes_p=2000
  Stage 2. MDP(G): n_episodes_g=2000
  Stage 3. MLP:    n_episodes_mlp=1000 (P,G 고정 → MLP만 업데이트)

[출력]
  output/exp_kim_dual_agent_10seed/
  ├── seed_{N}/stage1_mdp_p/, stage2_mdp_g/, stage3_mlp/
  │   └── val_ctr.npy, val_rewards.npy, rewards.npy, model_actor.pt
  ├── seed_{N}/eval_result.csv
  ├── exp_kim_dual_agent_10seed.log
  ├── all_seeds_summary.csv  (mean±std, min, max)
  ├── ctr_curve.png, reward_curve.png, final_eval_bar.png

[실행]
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_kim_dual_agent_10seed.py --gpu 0
  python experiments/exp_kim_dual_agent_10seed.py --eval_only --gpu 0
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
# GroupInfoExtractor 재사용 — fit(), get_cluster_centroid() 이미 검증됨
# GroupCentroidExtractor를 별도 구현하지 않고 기존 코드 그대로 사용
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


def get_group_state_91d(ext, state_91d: np.ndarray) -> np.ndarray:
    """
    GroupInfoExtractor.get_cluster_centroid() → 88D centroid
    → 91D로 패딩 (동적변수 3D = 0)
    MDP(G) DDPG 입력 차원(91D)에 맞춤.
    """
    c88 = ext.get_cluster_centroid(state_91d)        # (88,)
    return np.concatenate([c88, np.zeros(3)]).astype(np.float32)  # (91,)


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
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# =======================================================================
# [섹션 2] MLP — balance 자동 조절기
# =======================================================================

class BalanceMLP(nn.Module):
    """
    MDP(P)와 MDP(G)의 27D weight를 concat → 54D 입력
    → Linear(128) → ReLU → Linear(27) → 최종 추천 weight
    """
    def __init__(self, input_dim=54, hidden_dim=128, output_dim=27):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# =======================================================================
# [섹션 3] K-Means 집단 centroid 추출기
# =======================================================================

# GroupCentroidExtractor 제거 — GroupInfoExtractor.get_cluster_centroid() 사용
# =======================================================================
# [섹션 4] 공통 검증 루프 헬퍼
# =======================================================================

def _val_loop_p(agent, env, device, n=50):
    """MDP(P) 검증: 개인 state 직접 사용."""
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
    """MDP(G) 검증: centroid state 사용."""
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
    """MLP 검증: P+G+MLP 통합 평가."""
    val_r, val_s, cold_r, cold_n = 0, 0, 0, 0
    for _ in range(n):
        sv_raw = env.reset(); sc = 0
        while True:
            sp = torch.Tensor([sv_raw]).to(device)
            sg = torch.Tensor([get_group_state_91d(ext, sv_raw)]).to(device)
            with torch.no_grad():
                wf = mlp(torch.cat([actor_p(sp), actor_g(sg)], dim=1))
            av = np.clip(wf.cpu().numpy()[0],
                         env.action_space.low, env.action_space.high)
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
                    agent.update_parameters(
                        Transition(*zip(*memory.sample(bs))))
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
    logging.info(f"  [Stage 1 MDP(P) | Seed {seed}] 완료")
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
                    agent.update_parameters(
                        Transition(*zip(*memory.sample(bs))))
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
    logging.info(f"  [Stage 2 MDP(G) | Seed {seed}] 완료")
    env.close()
    return agent


# =======================================================================
# [섹션 7] Stage 3 — MLP 학습
# =======================================================================

def train_mlp(seed, n_episodes, device, agent_p, agent_g, ext, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"  [Stage 3 MLP | Seed {seed}] 시작")

    # P, G 완전 고정
    agent_p.actor.eval(); agent_g.actor.eval()

    mlp       = BalanceMLP(54, 128, 27).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 200, env)

    rewards, val_rewards, val_ctr = [], [], []

    for ep in range(n_episodes):
        sv_raw = env.reset(); ep_r = 0
        while True:
            sp = torch.Tensor([sv_raw]).to(device)
            sg = torch.Tensor([get_group_state_91d(ext, sv_raw)]).to(device)
            with torch.no_grad():
                wp = agent_p.actor(sp)
                wg = agent_g.actor(sg)
            wf = mlp(torch.cat([wp, wg], dim=1))   # gradient 있음
            av = np.clip(wf.detach().cpu().numpy()[0],
                         env.action_space.low, env.action_space.high)
            ns_raw, r, done, _ = env.step(av)

            # REINFORCE: -reward × mean(|w_final|)
            loss = -(torch.tensor(float(r), device=device) * wf.mean())
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            sv_raw = ns_raw; ep_r += r
            if done: break
        rewards.append(ep_r)

        if ep > 0 and ep % 20 == 0:
            vr, vs, cr, cn = _val_loop_mlp(
                agent_p.actor, agent_g.actor, mlp, env, device, ext)
            avg_r, ctr, _ = _log_val("MLP", ep, vr, vs, cr, cn)
            val_rewards.append(avg_r); val_ctr.append(ctr)

    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))
    torch.save(mlp.state_dict(),             f"{output_dir}/mlp.pt")
    logging.info(f"  [Stage 3 MLP | Seed {seed}] 완료")
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
    actor_p.load_state_dict(torch.load(
        f"{seed_dir}/stage1_mdp_p/model_actor.pt",
        map_location=device, weights_only=False))
    actor_p.eval()

    actor_g = Actor(128, obs_dim, aspace).to(device)
    actor_g.load_state_dict(torch.load(
        f"{seed_dir}/stage2_mdp_g/model_actor.pt",
        map_location=device, weights_only=False))
    actor_g.eval()

    mlp = BalanceMLP(54, 128, 27).to(device)
    mlp.load_state_dict(torch.load(
        f"{seed_dir}/stage3_mlp/mlp.pt",
        map_location=device, weights_only=False))
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
            av = np.clip(wf.cpu().numpy()[0],
                         env.action_space.low, env.action_space.high)
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
    with open(f"{seed_dir}/eval_result.csv", "w", newline="",
              encoding="utf-8-sig") as f:
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
            w.writerow({"model":"mean±std","seed":"─",
                        m:f"{np.mean(vals):.4f}±{np.std(vals):.4f}"})

    logging.info(f"요약 CSV 저장: {path}")
    print(f"\n{'='*65}")
    print("  김홍(2023) MDP(P)+G+MLP 정량적 검증 결과 (10 Seed)")
    print(f"{'='*65}")
    print(f"  {'Seed':<8}{'AvgReward':>11}{'CTR':>9}{'ColdCTR':>11}{'AvgSteps':>11}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['seed']:<8}{r['AvgReward']:>11.4f}{r['CTR']:>9.4f}"
              f"{r['ColdStartCTR']:>11.4f}{r['AvgSteps']:>11.2f}")
    print(f"  {'─'*50}")
    for m in ["AvgReward","CTR","ColdStartCTR"]:
        vals = [r[m] for r in results]
        print(f"  {'mean±std':<8}  {m}: {np.mean(vals):.4f}±{np.std(vals):.4f}"
              f"  [min={min(vals):.4f}, max={max(vals):.4f}]")
    print(f"{'='*65}\n")


def plot_learning_curves(seeds, output_dir):
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))
    for key, ylabel, fname in [
        ("val_ctr",     "CTR",       "ctr_curve.png"),
        ("val_rewards", "AvgReward", "reward_curve.png"),
    ]:
        curves = []
        fig, ax = plt.subplots(figsize=(11, 5))
        for i, seed in enumerate(seeds):
            p = f"{output_dir}/seed_{seed}/stage3_mlp/{key}.npy"
            if os.path.exists(p):
                arr = np.load(p); curves.append(arr)
                ax.plot(np.arange(len(arr))*20, arr,
                        color=colors[i], alpha=0.35, linewidth=1,
                        label=f"seed={seed}")
        if len(curves) >= 2:
            ml = min(len(c) for c in curves)
            A  = np.array([c[:ml] for c in curves])
            x  = np.arange(ml) * 20
            ax.plot(x, A.mean(0), color='black', linewidth=2.5,
                    label='Mean', zorder=5)
            ax.fill_between(x, A.mean(0)-A.std(0), A.mean(0)+A.std(0),
                            color='gray', alpha=0.25, label='±1 Std')
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"김홍(2023) MDP+G+MLP: {ylabel} (Stage3 MLP)", fontsize=12)
        ax.legend(fontsize=7, ncol=3); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{fname}", dpi=150); plt.close()
        logging.info(f"학습 곡선 저장: {output_dir}/{fname}")


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
        ax.axhline(mean_, color='purple', linewidth=1.8, linestyle='--',
                   label=f"Mean: {mean_:.4f}")
        ax.axhspan(mean_-std_, mean_+std_, alpha=0.12,
                   color='purple', label=f"±Std: {std_:.4f}")
        ax.set_title(metric, fontsize=11); ax.set_xlabel("Seed", fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_eval_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"최종 평가 그래프 저장: {output_dir}/final_eval_bar.png")


# =======================================================================
# [섹션 10] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="김홍(2023) 이중 에이전트 재현 — 10 Seed 정량 검증")
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
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(f"{args.output_dir}/exp_kim_dual_agent_10seed.log")])

    device = get_device_with_id(args.gpu)
    if torch.cuda.is_available():
        logging.info(f"  GPU: {torch.cuda.current_device()}번 — "
                     f"{torch.cuda.get_device_name(torch.cuda.current_device())}")

    logging.info("="*60)
    logging.info("김홍(2023) 이중 에이전트 재현 실험 시작")
    logging.info(f"  Seeds: {args.seeds}  (총 {len(args.seeds)}개)")
    logging.info(f"  Stage1 MDP(P): {args.n_episodes_p} ep | "
                 f"Stage2 MDP(G): {args.n_episodes_g} ep | "
                 f"Stage3 MLP: {args.n_episodes_mlp} ep")
    logging.info(f"  평가: {args.n_eval} ep | 디바이스: {device}")
    logging.info("="*60)

    # GroupInfoExtractor 재사용 — 이미 검증된 파싱 방식
    # get_cluster_centroid()가 88D state 기준으로 centroid(88D) 반환
    ext = GroupInfoExtractor(
        n_clusters=100,
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    ext.fit()

    all_results = []

    for seed in args.seeds:
        seed_dir = f"{args.output_dir}/seed_{seed}"
        os.makedirs(seed_dir, exist_ok=True)
        logging.info(f"\n{'='*50}\n[Seed {seed}] 3단계 학습\n{'='*50}")

        if not args.eval_only:
            agent_p = train_mdp_p(seed, args.n_episodes_p, device,
                                  f"{seed_dir}/stage1_mdp_p")
            agent_g = train_mdp_g(seed, args.n_episodes_g, device, ext,
                                  f"{seed_dir}/stage2_mdp_g")
            train_mlp(seed, args.n_episodes_mlp, device,
                      agent_p, agent_g, ext, f"{seed_dir}/stage3_mlp")

        if os.path.exists(f"{seed_dir}/stage3_mlp/mlp.pt"):
            result = evaluate_one_seed(seed, device, ext, seed_dir, args.n_eval)
            all_results.append(result)
        else:
            logging.warning(f"[Seed {seed}] MLP 모델 없음")

    if all_results:
        save_summary_csv(all_results, args.output_dir)
        plot_final_eval(all_results, args.output_dir)
    plot_learning_curves(args.seeds, args.output_dir)

    if all_results:
        logging.info("="*60)
        logging.info("김홍(2023) 최종 결과 (mean ± std)")
        logging.info("="*60)
        for m in ["AvgReward","CTR","ColdStartCTR","AvgSteps"]:
            vals = [r[m] for r in all_results]
            logging.info(f"  {m:<15}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
                         f"  (min={min(vals):.4f}, max={max(vals):.4f})")
        logging.info("="*60)

    logging.info(f"실험 완료. 결과: {args.output_dir}/")


if __name__ == "__main__":
    main()