"""
=======================================================================
experiments/exp_T1_frozen_10seed.py  —  T1 Frozen 정량적 검증 (10 Seed)
=======================================================================

[실험 목적]
  T1 조건 (확정 4개 속성)에서 DistilBERT를 Frozen시킨 모델의
  10 seed 반복실험 → mean ± std 형태의 신뢰할 수 있는 성능 수치 도출.

[이 실험이 가능하게 하는 비교 2가지]

  비교 (1): T1-FT (exp_T1_10seed) vs T1-Frozen (이 파일)
    - 통제: T1 속성 4개 동일
    - 독립변수: DistilBERT trainable 여부
    → "속성을 늘리지 않은 상태에서 LM 고정 유무가
       강화학습 수렴성에 미치는 순수한 영향력" 증명

  비교 (2): T1-Frozen (이 파일) vs T2-Step1-Frozen (t2_run_multiseed_frozen_v2)
    - 통제: 둘 다 DistilBERT Frozen
    - 독립변수: is_occupied 문맥 추가 여부
    → "동일한 안정적 구조에서 직업 문맥 추가가
       Cold-start 문제를 얼마나 해결했는지 순수 문맥 효과" 증명

[exp2_t1_frozen.py vs 이 파일 차이]
  exp2: seed=1 단일 고정, ColdStartCTR 없음, np.clip 없음
  이 파일:
    - 10 seed 반복 (0~9)
    - ColdStartCTR 추가 (다른 experiments와 동일)
    - np.clip 추가 (Action Clipping, 공정한 비교)
    - Buffer Warmup 추가 (WARMUP_STEPS=128)
    - mean ± std CSV + PNG 저장

[다른 experiments와 동일한 검증 조건]
  - n_episodes: 2000 (학습)
  - n_eval: 5000 (평가, 김홍과 동일)
  - 검증 주기: 20 에피소드마다 50 에피소드 greedy
  - 지표: AvgReward, CTR, ColdStartCTR, AvgSteps
  - 출력: all_seeds_summary.csv (mean ± std), ctr_curve.png, final_eval_bar.png

[파일 위치]
  virtual-taobao-rl/
  └── experiments/
      └── exp_T1_frozen_10seed.py  ← 이 파일

[실행]
  # 헤드 디렉토리(virtual-taobao-rl/)에서
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_T1_frozen_10seed.py --gpu 0
  python experiments/exp_T1_frozen_10seed.py --seeds 0 1 2 3 4 5 6 7 8 9 --gpu 1
  python experiments/exp_T1_frozen_10seed.py --eval_only --gpu 0  # 학습 완료 후 평가만
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
from tqdm import tqdm

# 헤드 디렉토리(virtual-taobao-rl/)에서 실행 기준
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gym
import virtualTB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from t1_textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder, LMActor, LMCritic


# =======================================================================
# [섹션 1] 공통 유틸리티
# =======================================================================

def get_device_with_id(gpu_id: Optional[int] = None) -> str:
    """
    GPU 번호 명시 지정 함수.
    exp_T1_10seed.py와 동일한 방식.
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            n_gpus = torch.cuda.device_count()
            if gpu_id >= n_gpus:
                logging.warning(f"GPU {gpu_id} 없음 (가용: 0~{n_gpus-1}). GPU 0 사용.")
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
            return f"cuda:{gpu_id}"
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_global_seed(seed: int, env=None):
    """RL 재현성을 위한 전역 seed 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


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
    """
    T1-Frozen 전용 OUNoise.
    sigma=0.1 (Frozen 상태에서 이미 어느 정도 수렴된 정책 → 정밀 탐색)
    T1-FT(sigma=0.2)보다 좁은 탐색 — Frozen 특성에 맞게 조정.
    """
    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.1):
        self.action_dim = action_dim
        self.scale  = scale
        self.mu     = mu
        self.theta  = theta
        self.sigma  = sigma
        self.state  = np.ones(action_dim) * mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# =======================================================================
# [섹션 2] T1-Frozen LMDDPG 에이전트
# =======================================================================

class LMDDPG_T1_Frozen:
    """
    T1 조건 + DistilBERT Frozen DDPG 에이전트.

    T1-FT(exp_T1_10seed)와의 유일한 차이:
      encoder = DistilBERTEncoder(trainable=False)  ← Frozen
      Actor FC LR: 3e-5  (FT의 1e-4보다 낮게 — FC만 업데이트되므로)
      Critic LR:   1e-4  (FT의 1e-3보다 낮게 — Q값 안정화)

    exp2_t1_frozen.py에서의 개선:
      + Action Clipping (np.clip)
      + Buffer Warmup (WARMUP_STEPS)
      + ColdStartCTR 측정
    """
    def __init__(self, gamma, tau, hidden_size, action_space, encoder, device):
        self.device = device
        self.gamma  = gamma
        self.tau    = tau

        self.actor        = LMActor(hidden_size, action_space, encoder).to(device)
        self.actor_target = LMActor(hidden_size, action_space, encoder).to(device)

        # Frozen이므로 encoder.parameters()는 실제로 업데이트되지 않음
        # (requires_grad=False → optimizer에 넣어도 step 시 변화 없음)
        # 구조 통일성을 위해 동일한 형태 유지
        encoder_param_ids = {id(p) for p in encoder.parameters()}
        actor_fc_params   = [p for p in self.actor.parameters()
                             if id(p) not in encoder_param_ids]
        self.actor_optim  = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": 1e-5},   # 무의미 (Frozen)
            {"params": actor_fc_params,       "lr": 3e-5},  # FC만 실제 업데이트
        ])

        self.critic        = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_target = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_optim  = torch.optim.Adam(
            self.critic.parameters(), lr=1e-4)  # Q값 안정화

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

    def load_model(self, path_prefix: str):
        self.actor.load_state_dict(
            torch.load(f"{path_prefix}_actor.pt",
                       map_location=self.device, weights_only=False))
        self.actor_target.load_state_dict(self.actor.state_dict())


# =======================================================================
# [섹션 3] 학습 루프
# =======================================================================

def train_one_seed(seed: int, n_episodes: int, device: str,
                   group_extractor: GroupInfoExtractor,
                   output_dir: str) -> dict:
    """
    단일 seed로 T1-Frozen 학습.
    exp2_t1_frozen.py 구조 기반 + 10 seed 확장을 위한 개선.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\n[T1-Frozen | Seed {seed}] 학습 시작")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)

    # ★ DistilBERT Frozen — exp2와 동일, trainable=False
    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)

    agent  = LMDDPG_T1_Frozen(
        gamma=0.7, tau=0.003, hidden_size=128,
        action_space=env.action_space,
        encoder=encoder, device=device,
    )
    memory  = ReplayMemory(10000)
    ounoise = OUNoise(env.action_space.shape[0])  # sigma=0.1 (Frozen 기본값)

    # Buffer Warmup: 초반 랜덤 경험으로 가중치 오염 방지
    WARMUP_STEPS = 32 * 4   # 128 step

    rewards, val_rewards, val_ctr = [], [], []
    total_steps = 0

    for i_ep in range(n_episodes):
        state_91d = env.reset()
        ounoise.reset()
        ep_reward = 0

        while True:
            res    = prepare_state(state_91d, group_extractor)
            prompt = res['prompt']
            dyn    = torch.FloatTensor(
                res['dynamic_numeric']).unsqueeze(0).to(device)

            action_t = agent.select_action(prompt, dyn)
            noise    = torch.FloatTensor(ounoise.noise()).to(device)
            action_t = (action_t + noise).clamp(-1, 1)

            # Action Clipping: 환경 허용 범위 이중 안전장치
            action_np = action_t.cpu().numpy().squeeze()
            action_np = np.clip(action_np,
                                env.action_space.low,
                                env.action_space.high)

            next_91d, reward, done, _ = env.step(action_np)

            memory.push(
                state_91d,
                action_t.detach().cpu().unsqueeze(0),
                torch.Tensor([not done]),
                next_91d,
                torch.Tensor([reward]),
            )
            state_91d   = next_91d
            ep_reward  += reward
            total_steps += 1

            # Buffer Warmup 후에만 업데이트
            if len(memory) > 32 and total_steps > WARMUP_STEPS:
                for _ in range(5):
                    batch = Transition(*zip(*memory.sample(32)))
                    agent.update_parameters(batch, group_extractor)

            if done:
                break

        rewards.append(ep_reward)
        logging.info(f"  [Seed {seed}] Ep {i_ep:4d} | Reward: {ep_reward:.1f}")

        # 검증 — 다른 experiments와 동일: 20 ep마다 50 ep greedy
        if i_ep > 0 and i_ep % 20 == 0:
            val_r, val_s, cold_r, cold_n = 0, 0, 0, 0

            for _ in range(50):
                s = env.reset()
                s_res = prepare_state(s, group_extractor)
                p = s_res['prompt']
                d = torch.FloatTensor(
                    s_res['dynamic_numeric']).unsqueeze(0).to(device)
                step_cnt = 0

                while True:
                    a  = agent.select_action(p, d).cpu().numpy().squeeze()
                    a  = np.clip(a, env.action_space.low, env.action_space.high)
                    ns, r, done, _ = env.step(a)

                    val_r += r; val_s += 1
                    if step_cnt == 0:          # Cold-start: 첫 번째 step
                        cold_r += r; cold_n += 1

                    nr = prepare_state(ns, group_extractor)
                    p  = nr['prompt']
                    d  = torch.FloatTensor(
                        nr['dynamic_numeric']).unsqueeze(0).to(device)
                    step_cnt += 1
                    if done: break

            avg_r    = val_r / 50
            ctr      = val_r / val_s / 10 if val_s > 0 else 0
            cold_ctr = cold_r / cold_n / 10 if cold_n > 0 else 0
            val_rewards.append(avg_r)
            val_ctr.append(ctr)

            logging.info(
                f"  [T1-Frozen | Seed {seed}][검증] Ep {i_ep:4d} | "
                f"AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | "
                f"ColdCTR: {cold_ctr:.4f}"
            )

    # 결과 저장
    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))
    agent.save_model(f"{output_dir}/model")
    logging.info(f"  [Seed {seed}] 학습 완료. 저장: {output_dir}/")

    env.close()
    return {"val_ctr": val_ctr, "val_rewards": val_rewards}


# =======================================================================
# [섹션 4] 평가 루프
# =======================================================================

def evaluate_one_seed(seed: int, device: str,
                      group_extractor: GroupInfoExtractor,
                      output_dir: str,
                      n_eval: int = 5000) -> dict:
    """
    학습 완료 모델을 greedy action으로 최종 평가.
    학습 seed와 다른 seed(seed+9999)로 평가하여 overfitting 방지.
    """
    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 9999, env)

    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)
    agent   = LMDDPG_T1_Frozen(
        gamma=0.7, tau=0.003, hidden_size=128,
        action_space=env.action_space,
        encoder=encoder, device=device,
    )
    agent.load_model(f"{output_dir}/model")
    agent.actor.eval()

    total_r, total_s, cold_r, cold_n = 0.0, 0, 0.0, 0

    for _ in tqdm(range(n_eval),
                  desc=f"  T1-Frozen 평가 | Seed {seed:<3}", leave=False):
        s = env.reset()
        step_cnt = 0

        while True:
            res = prepare_state(s, group_extractor)
            p   = res['prompt']
            d   = torch.FloatTensor(
                res['dynamic_numeric']).unsqueeze(0).to(device)
            a   = agent.select_action(p, d).cpu().numpy().squeeze()
            a   = np.clip(a, env.action_space.low, env.action_space.high)
            ns, r, done, _ = env.step(a)

            if step_cnt == 0:
                cold_r += r; cold_n += 1
            total_r += r; total_s += 1
            step_cnt += 1
            s = ns
            if done: break

    result = {
        "model":        "T1-Frozen",
        "seed":         seed,
        "AvgReward":    round(total_r / n_eval, 4),
        "CTR":          round(total_r / total_s / 10, 4) if total_s > 0 else 0,
        "ColdStartCTR": round(cold_r  / cold_n  / 10, 4) if cold_n  > 0 else 0,
        "AvgSteps":     round(total_s / n_eval, 2),
        "n_eval":       n_eval,
    }

    with open(f"{output_dir}/eval_result.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    logging.info(
        f"  [T1-Frozen | Seed {seed}] "
        f"AvgReward: {result['AvgReward']:.4f} | "
        f"CTR: {result['CTR']:.4f} | "
        f"ColdCTR: {result['ColdStartCTR']:.4f}"
    )
    env.close()
    return result


# =======================================================================
# [섹션 5] 결과 집계 및 시각화
# =======================================================================

def save_summary_csv(results: list, output_dir: str):
    """mean ± std 포함 CSV 저장 — 논문 Table 직접 활용."""
    path = f"{output_dir}/all_seeds_summary.csv"
    fields = ["model", "seed", "AvgReward", "CTR", "ColdStartCTR",
              "AvgSteps", "n_eval"]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
        # mean ± std 행
        for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[m] for r in results]
            writer.writerow({
                "model": "mean±std", "seed": "─",
                m: f"{np.mean(vals):.4f}±{np.std(vals):.4f}",
            })

    logging.info(f"요약 CSV 저장: {path}")

    # 콘솔 출력
    print(f"\n{'='*60}")
    print("  T1-Frozen 정량적 검증 결과 (10 Seed)")
    print(f"{'='*60}")
    print(f"  {'Seed':<8} {'AvgReward':>10} {'CTR':>8} {'ColdCTR':>10} {'AvgSteps':>10}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['seed']:<8} {r['AvgReward']:>10.4f} "
              f"{r['CTR']:>8.4f} {r['ColdStartCTR']:>10.4f} "
              f"{r['AvgSteps']:>10.2f}")
    print(f"  {'─'*50}")
    for m in ["AvgReward", "CTR", "ColdStartCTR"]:
        vals = [r[m] for r in results]
        print(f"  {'mean±std':<8}  {m}: "
              f"{np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"{'='*60}\n")


def plot_learning_curves(seeds: list, output_dir: str):
    """10 seed CTR 학습 곡선 — mean ± std 밴드 포함."""
    ctr_curves = []
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(seeds)))

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, seed in enumerate(seeds):
        p = f"{output_dir}/seed_{seed}/val_ctr.npy"
        if os.path.exists(p):
            arr = np.load(p)
            ctr_curves.append(arr)
            x = np.arange(len(arr)) * 20
            ax.plot(x, arr, color=colors[i], alpha=0.35,
                    linewidth=1, label=f"seed={seed}")

    if len(ctr_curves) >= 2:
        min_len = min(len(c) for c in ctr_curves)
        arr_all = np.array([c[:min_len] for c in ctr_curves])
        x_mean  = np.arange(min_len) * 20
        mean_   = arr_all.mean(0)
        std_    = arr_all.std(0)
        ax.plot(x_mean, mean_, color='darkgreen', linewidth=2.5,
                label='Mean', zorder=5)
        ax.fill_between(x_mean, mean_ - std_, mean_ + std_,
                        color='green', alpha=0.2, label='±1 Std')

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("CTR", fontsize=11)
    ax.set_title("T1-Frozen: CTR Learning Curves (10 Seeds)",
                 fontsize=12)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ctr_curve.png", dpi=150)
    plt.close()
    logging.info(f"학습 곡선 저장: {output_dir}/ctr_curve.png")


def plot_final_eval(results: list, output_dir: str):
    """seed별 최종 평가 막대 그래프 + mean ± std."""
    metrics = ["AvgReward", "CTR", "ColdStartCTR"]
    seeds   = [r["seed"] for r in results]
    colors  = plt.cm.Greens(np.linspace(0.4, 0.9, len(seeds)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("T1-Frozen: 10-Seed Final Evaluation (mean ± std)",
                 fontsize=12)

    for ax, metric in zip(axes, metrics):
        vals  = [r[metric] for r in results]
        mean_ = np.mean(vals)
        std_  = np.std(vals)

        ax.bar([f"s{s}" for s in seeds], vals, color=colors, alpha=0.85)
        ax.axhline(mean_, color='darkgreen', linewidth=1.8,
                   linestyle='--', label=f"Mean: {mean_:.4f}")
        ax.axhspan(mean_ - std_, mean_ + std_, alpha=0.12,
                   color='green', label=f"±Std: {std_:.4f}")

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("Seed", fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_eval_bar.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    logging.info(f"최종 평가 그래프 저장: {output_dir}/final_eval_bar.png")


# =======================================================================
# [섹션 6] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="T1-Frozen 정량적 검증 — 10 Seed 반복실험"
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int,
        default=list(range(10)),
        help="실험 seed 목록 (기본: 0~9)"
    )
    parser.add_argument("--n_episodes", type=int, default=2000,
                        help="학습 에피소드 수")
    parser.add_argument("--n_eval",     type=int, default=5000,
                        help="평가 에피소드 수 (김홍과 동일)")
    parser.add_argument("--eval_only",  action="store_true",
                        help="학습 건너뛰고 평가만 실행")
    parser.add_argument("--gpu",        type=int, default=None,
                        help="사용할 GPU 번호 (예: --gpu 0)")
    parser.add_argument(
        "--output_dir",
        default="output/exp_T1_frozen_10seed",
        help="결과 저장 디렉토리 (T1-FT 결과와 분리)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{args.output_dir}/exp_T1_frozen_10seed.log"),
        ],
    )

    device = get_device_with_id(args.gpu)

    if torch.cuda.is_available():
        gpu_idx  = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_idx)
        logging.info(f"  GPU: {gpu_idx}번 — {gpu_name}")

    logging.info("="*55)
    logging.info("T1-Frozen 정량적 검증 실험 시작")
    logging.info(f"  Seeds:      {args.seeds}  (총 {len(args.seeds)}개)")
    logging.info(f"  학습 ep:    {args.n_episodes}")
    logging.info(f"  평가 ep:    {args.n_eval}")
    logging.info(f"  저장 위치:  {args.output_dir}")
    logging.info(f"  디바이스:   {device}")
    logging.info(f"  trainable:  False (DistilBERT Frozen)")
    logging.info(f"  Actor FC LR: 3e-5 | Critic LR: 1e-4 | sigma: 0.1")
    logging.info("="*55)

    # K-Means: 모든 seed 공유 (T1-FT와 동일 kmeans 사용)
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    all_eval_results = []

    for seed in args.seeds:
        seed_dir = f"{args.output_dir}/seed_{seed}"

        if not args.eval_only:
            train_one_seed(
                seed=seed,
                n_episodes=args.n_episodes,
                device=device,
                group_extractor=group_extractor,
                output_dir=seed_dir,
            )

        model_path = f"{seed_dir}/model_actor.pt"
        if os.path.exists(model_path):
            result = evaluate_one_seed(
                seed=seed,
                device=device,
                group_extractor=group_extractor,
                output_dir=seed_dir,
                n_eval=args.n_eval,
            )
            all_eval_results.append(result)
        else:
            logging.warning(f"[Seed {seed}] 모델 없음: {model_path}")

    # 결과 집계 + 시각화
    if all_eval_results:
        save_summary_csv(all_eval_results, args.output_dir)
        plot_final_eval(all_eval_results, args.output_dir)

    plot_learning_curves(args.seeds, args.output_dir)

    # 최종 mean ± std 로그
    if all_eval_results:
        logging.info("="*55)
        logging.info("T1-Frozen 최종 결과 (mean ± std)")
        logging.info("="*55)
        for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[m] for r in all_eval_results]
            logging.info(
                f"  {m:<15}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
                f"  (min={min(vals):.4f}, max={max(vals):.4f})"
            )
        logging.info("="*55)

    logging.info(f"실험 완료. 결과: {args.output_dir}/")


if __name__ == "__main__":
    main()