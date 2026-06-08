"""
=======================================================================
t1_run_multiseed.py  —  다중 Seed 자동화 실험 스크립트 (T1 조건)
=======================================================================

[실험 목적]
  현재 t1_trainer_lm_ddpg.py는 seed=0 하나로만 고정되어 있어
  CTR=1.0 같은 비현실적 수치가 특정 seed의 운인지 실제 성능인지 알 수 없음.

  RL 논문의 표준: 3~5개 seed로 반복 실험 후 mean ± std 형태로 보고.
  본 스크립트는 여러 seed를 자동 순회하며 학습+평가를 수행하고
  결과를 CSV와 PNG 그래프로 저장함.

[RL에서 다중 seed가 필요한 이유]
  동일 알고리즘/하이퍼파라미터라도 seed에 따라:
  - 초기 네트워크 가중치가 달라짐 (Xavier 초기화 랜덤성)
  - 탐색 경로(OUNoise 순서)가 달라짐
  - 환경 샘플링 순서가 달라짐
  → 단일 seed 결과만으로는 "운이 좋은 것"이라는 반박을 피할 수 없음
 
[출력 파일 구조]
  output/multiseed/
  ├── seed_0/
  │   ├── val_ctr.npy          ← 학습 중 CTR 곡선
  │   ├── val_rewards.npy      ← 학습 중 AvgReward 곡선
  │   └── eval_result.csv      ← 최종 평가 수치
  ├── seed_1/ ...
  ├── seed_42/ ...
  ├── all_seeds_summary.csv    ← 전체 seed 결과 요약
  ├── ctr_curve.png            ← CTR 학습 곡선 비교
  └── reward_curve.png         ← Reward 학습 곡선 비교

[실행 방법]
  # 기본 실행 (seed 0, 1, 42 — 3개 seed)
  python t1_run_multiseed.py

  # seed 지정
  python t1_run_multiseed.py --seeds 0 1 2 42 100

  # 에피소드 수 지정
  python t1_run_multiseed.py --seeds 0 1 42 --n_episodes 2000

  # 평가만 (이미 학습된 모델 있을 때)
  python t1_run_multiseed.py --eval_only --seeds 0 1 42
=======================================================================
"""

import os
import sys
import csv
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional, List

import gym
import virtualTB
import matplotlib
matplotlib.use('Agg')   # GUI 없는 서버 환경 대응
import matplotlib.pyplot as plt

from t1_textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder, LMActor, LMCritic


# =======================================================================
# [섹션 1] 공통 유틸리티
# =======================================================================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int, env=None):
    """
    RL 실험의 재현성을 위한 전역 seed 설정.

    단순히 env.seed()만으로는 부족함:
    - PyTorch 초기화 랜덤성: torch.manual_seed()
    - numpy 샘플링: np.random.seed()
    - Python 내장 random: random.seed()
    - CUDA 연산: torch.cuda.manual_seed_all()
    모두 동일 seed로 설정해야 완전한 재현성 확보.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)
    logging.info(f"[Seed 설정] 전역 seed = {seed}")


Transition = namedtuple('Transition',
    ('state_91d', 'action', 'mask', 'next_state_91d', 'reward'))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Optional[Transition]] = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
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
# [섹션 2] LMDDPG 에이전트 (t1_trainer_lm_ddpg.py와 동일)
# =======================================================================

class LMDDPG:
    def __init__(self, gamma, tau, hidden_size, action_space, encoder, device):
        self.device = device
        self.gamma  = gamma
        self.tau    = tau

        self.actor        = LMActor(hidden_size, action_space, encoder).to(device)
        self.actor_target = LMActor(hidden_size, action_space, encoder).to(device)

        encoder_param_ids = {id(p) for p in encoder.parameters()}
        actor_fc_params   = [p for p in self.actor.parameters()
                             if id(p) not in encoder_param_ids]
        self.actor_optim  = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": 1e-5},
            {"params": actor_fc_params,       "lr": 1e-4},
        ])

        self.critic        = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_target = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, prompt, dynamic_numeric):
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(prompt, dynamic_numeric)
        self.actor.train()
        return mu.clamp(-1, 1)

    def update_parameters(self, batch, group_extractor):
        prompts, dynamics, next_prompts, next_dynamics = [], [], [], []
        for s, ns in zip(batch.state_91d, batch.next_state_91d):
            r  = prepare_state(s,  group_extractor)
            nr = prepare_state(ns, group_extractor)
            prompts.append(r['prompt'])
            dynamics.append(r['dynamic_numeric'])
            next_prompts.append(nr['prompt'])
            next_dynamics.append(nr['dynamic_numeric'])

        dyn_t      = torch.FloatTensor(np.array(dynamics)).to(self.device)
        next_dyn_t = torch.FloatTensor(np.array(next_dynamics)).to(self.device)
        action_t   = torch.cat(list(batch.action)).to(self.device)
        reward_t   = torch.cat(list(batch.reward)).unsqueeze(1).to(self.device)
        mask_t     = torch.cat(list(batch.mask)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_act = self.actor_target(next_prompts, next_dyn_t)
            next_q   = self.critic_target(next_prompts, next_dyn_t, next_act)
            target_q = reward_t + (self.gamma * mask_t * next_q)

        self.critic_optim.zero_grad()
        curr_q     = self.critic(prompts, dyn_t, action_t)
        value_loss = F.mse_loss(curr_q, target_q)
        value_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = -self.critic(
            prompts, dyn_t, self.actor(prompts, dyn_t)).mean()
        policy_loss.backward()
        self.actor_optim.step()

        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, path_prefix: str):
        os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
        torch.save(self.actor.state_dict(),  f"{path_prefix}_actor.pt")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pt")

    def load_model(self, path_prefix: str, device: str):
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor.pt",
                       map_location=device, weights_only=False))
        self.actor_target.load_state_dict(self.actor.state_dict())


# =======================================================================
# [섹션 3] 단일 seed 학습 함수
# =======================================================================

def train_one_seed(
    seed: int,
    n_episodes: int,
    device: str,
    group_extractor: GroupInfoExtractor,
    output_dir: str,
) -> dict:
    """
    단일 seed로 LM-DDPG 학습 수행 후 학습 곡선 저장.

    Args:
        seed          : 현재 실험 seed
        n_episodes    : 학습 에피소드 수
        device        : 실행 디바이스
        group_extractor: 공유 K-Means 추출기
        output_dir    : 이 seed의 결과 저장 디렉토리

    Returns:
        dict: val_ctr 리스트, val_rewards 리스트
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\n{'='*55}")
    logging.info(f"[Seed {seed}] 학습 시작 | {n_episodes} 에피소드")
    logging.info(f"{'='*55}")

    # ── 환경 + Seed 설정 ───────────────────────────────────────────
    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)   # ★ 핵심: 모든 난수를 seed로 통일

    # ── DistilBERT 인코더 (seed별로 새로 초기화) ──────────────────
    is_mac = (device == "mps")
    encoder = DistilBERTEncoder(
        trainable=not is_mac,
        device=device,
        max_length=128,
    )

    # ── 에이전트 초기화 ────────────────────────────────────────────
    # set_global_seed 이후 초기화 → 가중치 초기화 랜덤성도 seed 적용
    agent = LMDDPG(
        gamma=0.7,
        tau=0.003,
        hidden_size=128,
        action_space=env.action_space,
        encoder=encoder,
        device=device,
    )

    memory     = ReplayMemory(10000)
    ounoise    = OUNoise(env.action_space.shape[0])
    batch_size = 32

    rewards, val_rewards, val_ctr = [], [], []
    value_losses, policy_losses   = [], []

    # ── 학습 루프 ─────────────────────────────────────────────────
    for i_ep in range(n_episodes):
        state_91d = env.reset()
        ounoise.reset()
        episode_reward = 0

        while True:
            res    = prepare_state(state_91d, group_extractor)
            prompt = res['prompt']
            dyn    = torch.FloatTensor(
                res['dynamic_numeric']).unsqueeze(0).to(device)

            action_t = agent.select_action(prompt, dyn)
            noise    = torch.FloatTensor(ounoise.noise()).to(device)
            action_t = (action_t + noise).clamp(-1, 1)

            next_state_91d, reward, done, _ = env.step(
                action_t.cpu().numpy().squeeze())

            memory.push(
                state_91d,
                action_t.detach().cpu().unsqueeze(0),
                torch.Tensor([not done]),
                next_state_91d,
                torch.Tensor([reward]),
            )
            state_91d      = next_state_91d
            episode_reward += reward

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    vl, pl = agent.update_parameters(batch, group_extractor)
                    value_losses.append(vl)
                    policy_losses.append(pl)

            if done:
                break

        rewards.append(episode_reward)

        # ── 검증 (20 에피소드마다) ────────────────────────────────
        if i_ep > 0 and i_ep % 20 == 0:
            val_r, val_s = 0, 0
            cold_r, cold_n = 0, 0   # ColdStartCTR 별도 집계

            for _ in range(50):
                s = env.reset()
                s_res = prepare_state(s, group_extractor)
                p = s_res['prompt']
                d = torch.FloatTensor(
                    s_res['dynamic_numeric']).unsqueeze(0).to(device)
                step = 0

                while True:
                    a = agent.select_action(p, d).cpu().numpy().squeeze()
                    ns, r, done, _ = env.step(a)
                    val_r += r
                    val_s += 1

                    # Cold-start 첫 step 별도 기록
                    if step == 0:
                        cold_r += r
                        cold_n += 1

                    nr = prepare_state(ns, group_extractor)
                    p  = nr['prompt']
                    d  = torch.FloatTensor(
                        nr['dynamic_numeric']).unsqueeze(0).to(device)
                    step += 1
                    if done:
                        break

            avg_r    = val_r / 50
            ctr      = val_r / val_s / 10 if val_s > 0 else 0
            cold_ctr = cold_r / cold_n / 10 if cold_n > 0 else 0

            val_rewards.append(avg_r)
            val_ctr.append(ctr)
            logging.info(
                f"[Seed {seed}][검증] Ep {i_ep:4d} | "
                f"AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | "
                f"ColdCTR: {cold_ctr:.4f}"
            )

    # ── 결과 저장 ─────────────────────────────────────────────────
    np.save(f"{output_dir}/val_ctr.npy",      np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy",  np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",      np.array(rewards))
    agent.save_model(f"{output_dir}/model")
    logging.info(f"[Seed {seed}] 학습 완료. 결과 저장: {output_dir}/")

    env.close()
    return {"val_ctr": val_ctr, "val_rewards": val_rewards}


# =======================================================================
# [섹션 4] 단일 seed 최종 평가 함수 (5000 에피소드 greedy)
# =======================================================================

def evaluate_one_seed(
    seed: int,
    device: str,
    group_extractor: GroupInfoExtractor,
    output_dir: str,
    n_eval_episodes: int = 5000,
) -> dict:
    """
    학습 완료된 모델을 greedy action으로 최종 평가.
    김홍(2023) evaluate.py와 동일한 방식: noise 없음, 5000 에피소드.

    ColdStartCTR 측정 방식:
      각 에피소드의 첫 번째 step(step=0)에서의 reward를 별도 집계
      → "사용자가 처음 왔을 때 첫 추천의 CTR"
    """
    logging.info(f"[Seed {seed}] 최종 평가 시작 ({n_eval_episodes} 에피소드)")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 9999, env)   # 학습 seed와 다른 seed로 평가

    is_mac = (device == "mps")
    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)
    agent = LMDDPG(
        gamma=0.7, tau=0.003, hidden_size=128,
        action_space=env.action_space, encoder=encoder, device=device,
    )
    agent.load_model(f"{output_dir}/model", device)
    agent.actor.eval()

    total_r, total_s   = 0.0, 0
    cold_r,  cold_n    = 0.0, 0

    for _ in range(n_eval_episodes):
        s = env.reset()
        step = 0
        while True:
            res = prepare_state(s, group_extractor)
            p   = res['prompt']
            d   = torch.FloatTensor(
                res['dynamic_numeric']).unsqueeze(0).to(device)

            a = agent.select_action(p, d).cpu().numpy().squeeze()
            ns, r, done, _ = env.step(a)

            if step == 0:
                cold_r += r
                cold_n += 1

            total_r += r
            total_s += 1
            step    += 1
            s = ns
            if done:
                break

    avg_reward = total_r / n_eval_episodes
    ctr        = total_r / total_s / 10 if total_s > 0 else 0
    cold_ctr   = cold_r  / cold_n  / 10 if cold_n  > 0 else 0
    avg_steps  = total_s / n_eval_episodes

    result = {
        "seed":        seed,
        "AvgReward":   round(avg_reward, 4),
        "CTR":         round(ctr, 4),
        "ColdStartCTR":round(cold_ctr, 4),
        "AvgSteps":    round(avg_steps, 2),
        "n_eval":      n_eval_episodes,
    }

    # seed별 평가 결과 저장
    with open(f"{output_dir}/eval_result.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    logging.info(
        f"[Seed {seed}] 평가 완료 | "
        f"AvgReward: {avg_reward:.4f} | CTR: {ctr:.4f} | "
        f"ColdCTR: {cold_ctr:.4f}"
    )
    env.close()
    return result


# =======================================================================
# [섹션 5] 결과 집계 및 시각화
# =======================================================================

def save_summary_csv(all_results: list, save_dir: str):
    """
    전체 seed 결과를 하나의 CSV로 저장.
    mean ± std 행도 추가하여 논문 표에 바로 사용 가능.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = f"{save_dir}/all_seeds_summary.csv"
    fields = ["seed", "AvgReward", "CTR", "ColdStartCTR", "AvgSteps", "n_eval"]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

        # mean ± std 행 추가
        for metric in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[metric] for r in all_results]
            writer.writerow({
                "seed":     f"mean±std",
                metric:     f"{np.mean(vals):.4f}±{np.std(vals):.4f}",
            })

    logging.info(f"전체 요약 CSV 저장: {path}")
    print(f"\n{'='*55}")
    print("  다중 Seed 실험 결과 요약")
    print(f"{'='*55}")
    print(f"  {'Seed':<8} {'AvgReward':>10} {'CTR':>8} {'ColdCTR':>10} {'AvgSteps':>10}")
    print(f"  {'-'*48}")
    for r in all_results:
        print(f"  {r['seed']:<8} {r['AvgReward']:>10.4f} "
              f"{r['CTR']:>8.4f} {r['ColdStartCTR']:>10.4f} "
              f"{r['AvgSteps']:>10.2f}")
    print(f"  {'-'*48}")
    for metric in ["AvgReward", "CTR", "ColdStartCTR"]:
        vals = [r[metric] for r in all_results]
        print(f"  {'mean±std':<8}  {metric}: "
              f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*55}\n")


def plot_learning_curves(seeds: list, save_dir: str):
    """
    각 seed의 학습 곡선(CTR, AvgReward)을 비교 그래프로 저장.

    - 개별 seed: 투명한 색선
    - 평균값:    굵은 실선
    - 표준편차:  음영 밴드
    """
    ctr_curves     = []
    reward_curves  = []

    for seed in seeds:
        seed_dir = f"{save_dir}/seed_{seed}"
        ctr_path    = f"{seed_dir}/val_ctr.npy"
        reward_path = f"{seed_dir}/val_rewards.npy"

        if os.path.exists(ctr_path):
            ctr_curves.append(np.load(ctr_path))
        if os.path.exists(reward_path):
            reward_curves.append(np.load(reward_path))

    if not ctr_curves:
        logging.warning("시각화할 학습 곡선 데이터 없음.")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    # ── CTR 학습 곡선 ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    min_len = min(len(c) for c in ctr_curves)
    ctr_arr = np.array([c[:min_len] for c in ctr_curves])
    x = np.arange(min_len) * 20   # 20 에피소드 간격

    for i, (seed, curve) in enumerate(zip(seeds, ctr_curves)):
        ax.plot(x[:len(curve[:min_len])], curve[:min_len],
                color=colors[i], alpha=0.4, linewidth=1,
                label=f"seed={seed}")

    mean_ctr = ctr_arr.mean(axis=0)
    std_ctr  = ctr_arr.std(axis=0)
    ax.plot(x, mean_ctr, color='black', linewidth=2.5,
            label='Mean', zorder=5)
    ax.fill_between(x, mean_ctr - std_ctr, mean_ctr + std_ctr,
                    color='gray', alpha=0.25, label='±1 Std')

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("CTR", fontsize=12)
    ax.set_title("T1 LM-DDPG: CTR Learning Curve (Multi-Seed)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/ctr_curve.png", dpi=150)
    plt.close()
    logging.info(f"CTR 곡선 저장: {save_dir}/ctr_curve.png")

    # ── AvgReward 학습 곡선 ──────────────────────────────────────
    if reward_curves:
        fig, ax = plt.subplots(figsize=(10, 5))
        min_len_r  = min(len(r) for r in reward_curves)
        reward_arr = np.array([r[:min_len_r] for r in reward_curves])

        for i, (seed, curve) in enumerate(zip(seeds, reward_curves)):
            ax.plot(x[:len(curve[:min_len_r])], curve[:min_len_r],
                    color=colors[i], alpha=0.4, linewidth=1,
                    label=f"seed={seed}")

        mean_r = reward_arr.mean(axis=0)
        std_r  = reward_arr.std(axis=0)
        ax.plot(x[:min_len_r], mean_r, color='black', linewidth=2.5,
                label='Mean', zorder=5)
        ax.fill_between(x[:min_len_r], mean_r - std_r, mean_r + std_r,
                        color='gray', alpha=0.25, label='±1 Std')

        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("AvgReward", fontsize=12)
        ax.set_title("T1 LM-DDPG: AvgReward Learning Curve (Multi-Seed)", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reward_curve.png", dpi=150)
        plt.close()
        logging.info(f"AvgReward 곡선 저장: {save_dir}/reward_curve.png")

    # ── ColdStartCTR 박스플롯 (최종 평가 기반) ───────────────────
    eval_csvs = []
    for seed in seeds:
        p = f"{save_dir}/seed_{seed}/eval_result.csv"
        if os.path.exists(p):
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    eval_csvs.append(row)

    if eval_csvs:
        metrics = ["AvgReward", "CTR", "ColdStartCTR"]
        vals    = {m: [float(r[m]) for r in eval_csvs] for m in metrics}

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, metric in zip(axes, metrics):
            ax.bar(range(len(seeds)),
                   vals[metric],
                   color=[colors[i] for i in range(len(seeds))],
                   alpha=0.8)
            ax.axhline(np.mean(vals[metric]),
                       color='black', linestyle='--', linewidth=1.5,
                       label=f"Mean={np.mean(vals[metric]):.3f}")
            ax.set_xticks(range(len(seeds)))
            ax.set_xticklabels([f"s={s}" for s in seeds], fontsize=9)
            ax.set_title(metric, fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle("T1 LM-DDPG: Final Evaluation (Multi-Seed)",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/final_eval_bar.png", dpi=150,
                    bbox_inches='tight')
        plt.close()
        logging.info(f"최종 평가 막대 그래프 저장: {save_dir}/final_eval_bar.png")


# =======================================================================
# [섹션 6] 메인 — 전체 seed 자동 순회
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="T1 LM-DDPG 다중 Seed 실험")
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 1, 42],
                        help="실험할 seed 목록 (기본: 0 1 42)")
    parser.add_argument("--n_episodes", type=int, default=2000,
                        help="학습 에피소드 수 (기본: 2000)")
    parser.add_argument("--n_eval", type=int, default=5000,
                        help="최종 평가 에피소드 수 (기본: 5000, 김홍 동일)")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습 건너뛰고 평가만 실행")
    parser.add_argument("--output_dir", default="output/multiseed",
                        help="결과 저장 루트 디렉토리")
    args = parser.parse_args()

    # ── 로깅 설정 ─────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/multiseed.log"),
        ],
    )

    device = get_device()
    logging.info(f"디바이스: {device}")
    logging.info(f"실험 Seeds: {args.seeds}")
    logging.info(f"학습 에피소드: {args.n_episodes} | 평가 에피소드: {args.n_eval}")

    # ── K-Means는 모든 seed가 공유 (데이터 기반, seed 무관) ───────
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    # ── seed별 순회 실험 ──────────────────────────────────────────
    all_eval_results = []

    for seed in args.seeds:
        seed_dir = f"{args.output_dir}/seed_{seed}"

        # 학습
        if not args.eval_only:
            train_one_seed(
                seed=seed,
                n_episodes=args.n_episodes,
                device=device,
                group_extractor=group_extractor,
                output_dir=seed_dir,
            )
        else:
            logging.info(f"[Seed {seed}] --eval_only: 학습 건너뜀")

        # 최종 평가
        model_path = f"{seed_dir}/model_actor.pt"
        if os.path.exists(model_path):
            result = evaluate_one_seed(
                seed=seed,
                device=device,
                group_extractor=group_extractor,
                output_dir=seed_dir,
                n_eval_episodes=args.n_eval,
            )
            all_eval_results.append(result)
        else:
            logging.warning(f"[Seed {seed}] 모델 파일 없음: {model_path}")

    # ── 전체 결과 집계 + 시각화 ───────────────────────────────────
    if all_eval_results:
        save_summary_csv(all_eval_results, args.output_dir)

    plot_learning_curves(args.seeds, args.output_dir)

    logging.info("\n다중 Seed 실험 완료.")
    logging.info(f"결과 위치: {args.output_dir}/")


if __name__ == "__main__":
    main()