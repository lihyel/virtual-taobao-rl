"""
=======================================================================
experiments/exp_t0_10seed.py  —  T0 수치 DDPG 정량적 검증 (10 Seed)
=======================================================================

[실험 목적]
  T0 수치 DDPG(수치형 모델)의 통계적으로 신뢰할 수 있는 성능 수치 도출.
  10개 seed × 학습 2000ep + 평가 5000ep → mean ± std 형태로 보고.

[T0 수치 DDPG 정의]
  - 구조: 91D 수치 벡터 → Linear(91→128) → Linear(128→27) + Tanh
  - 하이퍼파라미터: tau=0.02, gamma=0.7 (김홍 MDP(P)와 동일)
  - LM/DistilBERT 없음 — 순수 수치 DDPG
  - exp3_t0_baseline_v2.py의 NumericalDDPG, Actor, Critic, LayerNorm 재사용

[exp3_t0_baseline_v2.py와의 관계]
  - exp3_t0_baseline_v2.py: seed=0 단일 실험용 (김홍 재현)
  - 이 파일: 10 seed 반복 실험 + mean ± std 도출 전용
  - 핵심 클래스(NumericalDDPG, Actor, Critic, LayerNorm) 직접 import

[출력 파일]
  output/exp_t0_10seed/
  ├── seed_{N}/
  │   ├── val_ctr.npy
  │   ├── val_rewards.npy
  │   ├── rewards.npy
  │   ├── model_actor       ← 최종 Actor 가중치
  │   └── eval_result.csv
  ├── exp_t0_10seed.log
  ├── all_seeds_summary.csv ← mean ± std 포함 (논문 Table용)
  ├── ctr_curve.png         ← 10개 seed 학습 곡선 비교
  └── final_eval_bar.png    ← seed별 최종 평가 막대 그래프

[실행]
  python experiments/exp_t0_10seed.py --gpu 2
  python experiments/exp_t0_10seed.py --seeds 0 1 2 3 4 5 6 7 8 9 --gpu 2
  python experiments/exp_t0_10seed.py --seeds 1 2 3 4 5 6 7 8 9 --gpu 2
  python experiments/exp_t0_10seed.py --eval_only --gpu 2  # 학습 완료 후 평가만
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

# ── exp3_t0_baseline_v2.py의 클래스를 직접 import ────────────────────
from exp3_t0_baseline_v2 import (
    LayerNorm,
    Actor,
    Critic,
    NumericalDDPG,
    soft_update,
    hard_update,
)

# =======================================================================
# [T1 동일 조건 — 24D 추출]
# =======================================================================
# T1 제안 모델이 실제 사용하는 인덱스만 추출.
# 마스킹(0으로 채운 91D)이 아닌 완전 제거 → T1과 동등한 정보 조건.
# 네트워크 입력: 91D → 24D

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

def get_device_with_id(gpu_id=None) -> str:
    """GPU 번호를 명시적으로 지정할 수 있는 device 선택 함수."""
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
    """RL 실험의 재현성을 위한 전역 seed 설정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


Transition = namedtuple('Transition',
    ('state', 'action', 'mask', 'next_state', 'reward'))


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
    """김홍 trainer_MDP_P_.py, exp3_t0_baseline_v2.py와 동일."""
    def __init__(self, action_dimension, scale=0.1, mu=0,
                 theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale  = scale; self.mu = mu
        self.theta  = theta; self.sigma = sigma
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
# [섹션 2] 학습 루프 (exp3_t0_baseline_v2.py의 main()과 동일 구조)
# =======================================================================

def train_one_seed(seed: int, n_episodes: int, device: str,
                   output_dir: str) -> dict:
    """
    단일 seed로 T0 수치 DDPG 학습.
    exp3_t0_baseline_v2.py의 main()과 완전히 동일한 구조.
    seed만 인자로 받도록 일반화.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\n[T0 수치 DDPG | Seed {seed}] 학습 시작")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)      # ★ 핵심: seed를 인자로 받음

    agent = NumericalDDPG(
        gamma       = 0.7,
        tau         = 0.02,         # 김홍 MDP(P)와 동일
        hidden_size = 128,
        num_inputs  = T1_INPUT_DIM,                     # 24 (T1 동일 조건)
        action_space= env.action_space,
        device      = device,
    )

    memory     = ReplayMemory(10000)
    ounoise    = OUNoise(env.action_space.shape[0])
    batch_size = 32

    rewards, val_rewards, val_ctr = [], [], []
    value_losses, policy_losses   = [], []

    for i_episode in range(n_episodes):
        # 김홍 trainer_MDP_P_.py와 동일한 상태 처리
        # ★ T1 동일 조건: 24개 인덱스만 추출
        state = torch.Tensor([extract_t1_features(env.reset())]).to(device)
        episode_reward = 0

        while True:
            action = agent.select_action(state, ounoise).to("cpu")

            next_state_raw, reward, done, _ = env.step(action.numpy()[0])

            action_t   = torch.Tensor(action).to(device)
            mask       = torch.Tensor([not done]).to(device)
            # ★ T1 동일 조건: next_state도 24D 추출
            next_state = torch.Tensor([extract_t1_features(next_state_raw)]).to(device)
            reward_t   = torch.Tensor([reward]).to(device)

            memory.push(state, action_t, mask, next_state, reward_t)
            state         = next_state
            episode_reward += reward

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch       = Transition(*zip(*transitions))
                    vl, pl      = agent.update_parameters(batch)
                    value_losses.append(vl)
                    policy_losses.append(pl)

            if done:
                break

        rewards.append(episode_reward)
        logging.info(f"  [Ep {i_episode:4d}] Reward: {episode_reward:.1f}")

        # 검증 (김홍과 동일: 20 episode마다 50개 에피소드)
        if i_episode > 0 and i_episode % 20 == 0:
            episode_reward_val = 0
            episode_step       = 0

            for _ in range(50):
                # ★ T1 동일 조건: 검증 루프도 24D 추출
                state_v = torch.Tensor([extract_t1_features(env.reset())]).to(device)
                while True:
                    action_v = agent.select_action(state_v).cpu()
                    ns_raw, r, done, _ = env.step(action_v.numpy()[0])
                    episode_reward_val += r
                    episode_step       += 1
                    state_v = torch.Tensor([extract_t1_features(ns_raw)]).to(device)
                    if done: break

            avg_r = episode_reward_val / 50
            ctr   = episode_reward_val / episode_step / 10
            val_rewards.append(avg_r)
            val_ctr.append(ctr)
            vl = value_losses[-1] if value_losses else 0
            pl = policy_losses[-1] if policy_losses else 0
            logging.info(
                f"  [Seed {seed}][검증] Ep {i_episode:4d} | "
                f"AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | "
                f"VLoss: {vl:.4f} | PLoss: {pl:.4f}"
            )

    # 결과 저장
    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))

    # Actor 모델 저장 (평가 시 재사용)
    os.makedirs("models", exist_ok=True)
    torch.save(agent.actor.state_dict(), f"{output_dir}/model_actor.pt")
    logging.info(f"  [Seed {seed}] 학습 완료. Actor 저장: {output_dir}/model_actor.pt")

    env.close()
    return {"val_ctr": val_ctr, "val_rewards": val_rewards}


# =======================================================================
# [섹션 3] 평가 루프 (evaluate_benchmark_v2.py의 benchmark_t0_numerical 재사용)
# =======================================================================

class _LayerNorm(nn.Module):
    """
    evaluate_benchmark_v2.py의 _LayerNorm과 동일.
    exp3_t0_baseline_v2.py와 LayerNorm 방식 일치 확인됨 (view 기반).
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.affine = affine
        self.eps    = eps
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean  = x.view(x.size(0), -1).mean(1).view(*shape)
        std   = x.view(x.size(0), -1).std(1).view(*shape)
        y     = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class _NumericalActor(nn.Module):
    """
    평가 시 사용하는 Actor 구조.

    ★ exp3_t0_baseline_v2.py의 Actor와 레이어 이름 및 구조 완전히 동일:
      linear1(91→128) + LayerNorm + ReLU
      linear2(128→128) + LayerNorm + ReLU
      mu(128→27) + Tanh

    저장 키: "linear1.weight", "linear2.weight", "mu.weight" 등
    → load_state_dict 시 키 불일치 없음
    """
    def __init__(self, hidden_size=128, num_inputs=T1_INPUT_DIM, action_space=None):
        # ★ T1 동일 조건: 기본 입력 24D
        super().__init__()
        num_outputs = action_space.shape[0] if action_space else 27
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1     = _LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2     = _LayerNorm(hidden_size)
        self.mu      = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        return torch.tanh(self.mu(x))


def evaluate_one_seed(seed: int, device: str, output_dir: str,
                      n_eval: int = 5000) -> dict:
    """
    학습 완료된 T0 Actor를 로드하여 greedy action으로 최종 평가.
    evaluate_benchmark_v2.py의 benchmark_t0_numerical()과 동일 방식.
    """
    actor = _NumericalActor(
        hidden_size=128,
        num_inputs=T1_INPUT_DIM,   # ★ 24D (학습 시와 동일해야 load_state_dict 성공)
        action_space=None,         # num_outputs=27 기본값 사용
    ).to(device)

    actor_path = f"{output_dir}/model_actor.pt"
    actor.load_state_dict(
        torch.load(actor_path, map_location=device, weights_only=False)
    )
    actor.eval()

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 9999, env)   # 학습 seed와 다른 seed로 평가

    total_r, total_s, cold_r, cold_n = 0.0, 0, 0.0, 0

    for _ in tqdm(range(n_eval), desc=f"  T0 평가 | Seed {seed:<3}", leave=False):
        # 김홍 evaluate.py의 infer_mdp_p()와 동일
        # ★ T1 동일 조건: 24개 인덱스만 추출
        state = torch.Tensor([extract_t1_features(env.reset())]).to(device)
        step_cnt = 0

        while True:
            with torch.no_grad():
                action = actor(state).cpu()
            ns_raw, r, done, _ = env.step(action.numpy()[0])

            if step_cnt == 0:
                cold_r += r; cold_n += 1
            total_r += r; total_s += 1
            step_cnt += 1
            # ★ T1 동일 조건: next_state도 24D 추출
            state = torch.Tensor([extract_t1_features(ns_raw)]).to(device)
            if done: break

    result = {
        "model":        "T0 수치 DDPG",
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
        f"  [T0 수치 DDPG | Seed {seed}] "
        f"AvgReward: {result['AvgReward']:.4f} | "
        f"CTR: {result['CTR']:.4f} | "
        f"ColdCTR: {result['ColdStartCTR']:.4f}"
    )
    env.close()
    return result


# =======================================================================
# [섹션 4] 결과 집계 및 시각화
# =======================================================================

def save_summary_csv(results: list, output_dir: str):
    path = f"{output_dir}/all_seeds_summary.csv"
    # ── 개별 seed 행 필드 ──────────────────────────────────────────────
    fields = ["model", "seed", "AvgReward", "CTR", "ColdStartCTR",
              "AvgSteps", "n_eval"]
    # ── 통계 요약 행 필드 (mean, std, min, max) ────────────────────────
    stat_fields = ["model", "seed",
                   "AvgReward_mean", "AvgReward_std", "AvgReward_min", "AvgReward_max",
                   "CTR_mean",       "CTR_std",       "CTR_min",       "CTR_max",
                   "ColdStartCTR_mean", "ColdStartCTR_std",
                   "ColdStartCTR_min",  "ColdStartCTR_max",
                   "AvgSteps_mean",  "AvgSteps_std",  "AvgSteps_min",  "AvgSteps_max",
                   "n_seeds"]
    all_fields = list(dict.fromkeys(fields + stat_fields))  # 중복 제거, 순서 유지

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()

        # 개별 seed 결과 행
        writer.writerows(results)

        # 구분선 역할의 빈 행
        writer.writerow({"model": "────", "seed": "────"})

        # ── 통계 요약 행 (mean ± std, min, max 모두 포함) ───────────────
        stat_row = {"model": "statistics", "seed": f"n={len(results)}",
                    "n_seeds": len(results)}
        for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[m] for r in results]
            stat_row[f"{m}_mean"] = round(np.mean(vals), 4)
            stat_row[f"{m}_std"]  = round(np.std(vals),  4)
            stat_row[f"{m}_min"]  = round(min(vals),      4)
            stat_row[f"{m}_max"]  = round(max(vals),      4)
        writer.writerow(stat_row)

        # 기존 호환성: mean±std 형식 행도 추가 (논문 Table 바로 복사용)
        writer.writerow({"model": "────", "seed": "────"})
        for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[m] for r in results]
            writer.writerow({
                "model": "mean±std",
                "seed":  "─",
                m:       f"{np.mean(vals):.4f}±{np.std(vals):.4f}",
            })

    logging.info(f"요약 CSV 저장: {path}")

    # 콘솔 출력
    print(f"\n{'='*60}")
    print("  T0 수치 DDPG 정량적 검증 결과 (10 Seed)")
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
              f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
              f"  [min={min(vals):.4f}, max={max(vals):.4f}]")
    print(f"{'='*60}\n")


def plot_learning_curves(seeds: list, output_dir: str):
    """
    10개 seed의 학습 곡선 비교 (mean ± std 밴드 포함).
    CTR + AvgReward 두 개 PNG 저장.
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    for metric_key, ylabel, title, filename in [
        ("val_ctr",     "CTR",       "CTR Learning Curves (10 Seeds)",        "ctr_curve.png"),
        ("val_rewards", "AvgReward", "AvgReward Learning Curves (10 Seeds)",  "reward_curve.png"),
    ]:
        curves = []
        fig, ax = plt.subplots(figsize=(11, 5))

        for i, seed in enumerate(seeds):
            p = f"{output_dir}/seed_{seed}/{metric_key}.npy"
            if os.path.exists(p):
                arr = np.load(p)
                curves.append(arr)
                x = np.arange(len(arr)) * 20
                ax.plot(x, arr, color=colors[i], alpha=0.35,
                        linewidth=1, label=f"seed={seed}")

        if len(curves) >= 2:
            min_len = min(len(c) for c in curves)
            arr_all = np.array([c[:min_len] for c in curves])
            x_mean  = np.arange(min_len) * 20
            mean_   = arr_all.mean(0)
            std_    = arr_all.std(0)
            ax.plot(x_mean, mean_, color='black', linewidth=2.5,
                    label='Mean', zorder=5)
            ax.fill_between(x_mean, mean_ - std_, mean_ + std_,
                            color='gray', alpha=0.25, label='±1 Std')
        elif len(curves) == 1:
            x = np.arange(len(curves[0])) * 20
            ax.plot(x, curves[0], color='black', linewidth=2.0, label='seed 0')

        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"T0 Numerical DDPG: {title}", fontsize=12)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f"{output_dir}/{filename}"
        plt.savefig(save_path, dpi=150)
        plt.close()
        logging.info(f"학습 곡선 저장: {save_path}")


def plot_final_eval(results: list, output_dir: str):
    """seed별 최종 평가 결과 막대 그래프."""
    metrics = ["AvgReward", "CTR", "ColdStartCTR"]
    seeds   = [r["seed"] for r in results]
    colors  = plt.cm.Oranges(np.linspace(0.4, 0.9, len(seeds)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("T0 Numerical DDPG: 10-Seed Final Evaluation",
                 fontsize=12)

    for ax, metric in zip(axes, metrics):
        vals  = [r[metric] for r in results]
        mean_ = np.mean(vals)
        std_  = np.std(vals)

        ax.bar([f"s{s}" for s in seeds], vals, color=colors, alpha=0.85)
        ax.axhline(mean_, color='red', linewidth=1.8, linestyle='--',
                   label=f"Mean: {mean_:.4f}")
        ax.axhspan(mean_ - std_, mean_ + std_, alpha=0.12,
                   color='red', label=f"±Std: {std_:.4f}")

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
# [섹션 5] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="T0 수치 DDPG 정량적 검증 — 10 Seed 반복실험"
    )
    parser.add_argument("--seeds",      nargs="+", type=int,
                        default=list(range(10)),
                        help="실험 seed 목록 (기본: 0~9)")
    parser.add_argument("--n_episodes", type=int, default=2000,
                        help="학습 에피소드 수 (기본: 2000, 김홍과 동일)")
    parser.add_argument("--n_eval",     type=int, default=5000,
                        help="평가 에피소드 수 (기본: 5000, 김홍과 동일)")
    parser.add_argument("--eval_only",  action="store_true",
                        help="학습 건너뛰고 평가만 실행")
    parser.add_argument("--gpu",        type=int, default=None,
                        help="사용할 GPU 번호 (예: --gpu 1)")
    parser.add_argument("--output_dir", default="output/exp_t0_10seed_24d")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/exp_t0_10seed_24d.log"),
        ],
    )

    device = get_device_with_id(args.gpu)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        gpu_idx  = torch.cuda.current_device()
        logging.info(f"  GPU: {gpu_idx}번 — {gpu_name}")

    logging.info("="*55)
    logging.info("T0 수치 DDPG 정량적 검증 실험 시작")
    logging.info(f"  Seeds:     {args.seeds}  (총 {len(args.seeds)}개)")
    logging.info(f"  학습 ep:   {args.n_episodes}")
    logging.info(f"  평가 ep:   {args.n_eval}")
    logging.info(f"  저장 위치: {args.output_dir}")
    logging.info(f"  디바이스:  {device}")
    logging.info("="*55)

    all_eval_results = []

    for seed in args.seeds:
        seed_dir = f"{args.output_dir}/seed_{seed}"

        if not args.eval_only:
            train_one_seed(
                seed=seed, n_episodes=args.n_episodes,
                device=device, output_dir=seed_dir,
            )

        actor_path = f"{seed_dir}/model_actor.pt"
        if os.path.exists(actor_path):
            result = evaluate_one_seed(
                seed=seed, device=device,
                output_dir=seed_dir,
                n_eval=args.n_eval,
            )
            all_eval_results.append(result)
        else:
            logging.warning(f"[Seed {seed}] 모델 없음: {actor_path}")

    # ── 결과 집계 + 시각화 ────────────────────────────────────────────
    if all_eval_results:
        save_summary_csv(all_eval_results, args.output_dir)
        plot_final_eval(all_eval_results, args.output_dir)

    plot_learning_curves(args.seeds, args.output_dir)

    # ── 최종 mean ± std 로그 ──────────────────────────────────────────
    if all_eval_results:
        logging.info("="*55)
        logging.info("T0 수치 DDPG 최종 결과 (mean ± std)")
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