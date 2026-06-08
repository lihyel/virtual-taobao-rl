"""
=======================================================================
experiments/exp_dynamic_ablation_frozen.py  —  동적변수 유효성 검증 (T1 Frozen 기반)
=======================================================================

[실험 목적]
  교수님 피드백:
    "768D DistilBERT vs 3D 동적변수 → 0.5% 미만의 동적 정보가
     실제로 강화학습에 변화를 주었는지 실험적 증명 필요"

  소거 실험 (Ablation Study):
    조건 A (정적만):  771D → 동적 3D 제거 → 768D (DistilBERT만)
    조건 B (정적+동적): 771D 전체 사용 (현재 T1 구조)

  증명 목표:
    - 조건 B의 CTR이 조건 A보다 높으면
      → "3D 동적정보가 의사결정에 기여함" 증명
    - 특히 Cold-start 이후 구간(step>0)에서 ColdCTR 비교
      → "실시간 클릭 반영(feat88)이 추천 개선에 기여함" 증명

[두 조건의 구조 차이]
  조건 A (Static-Only):
    env(91D) → textify → DistilBERT(768D) → concat(768D+0) → DDPG → 27D
    HybridStateEncoder에서 dynamic_numeric을 zeros(3)으로 대체

  조건 B (Static+Dynamic = 현재 T1):
    env(91D) → textify → DistilBERT(768D) → concat(768D+3D) → DDPG → 27D

[출력 파일]
  output/exp_dynamic_ablation_frozen/
  ├── static_only/              ← 조건 A
  │   ├── seed_{N}/eval_result.csv
  │   └── summary.csv
  ├── static_dynamic/           ← 조건 B (T1 기준)
  │   ├── seed_{N}/eval_result.csv
  │   └── summary.csv
  ├── exp_dynamic_ablation_frozen.log
  ├── ablation_comparison.csv   ← A vs B 비교표
  └── ablation_comparison.png   ← 시각화

[실행]
  python experiments/exp_dynamic_ablation_frozen.py
  python experiments/exp_dynamic_ablation_frozen.py --seeds 0 1 2 3 4 5
  python experiments/exp_dynamic_ablation_frozen.py --eval_only
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gym
import virtualTB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from t1_textify import (
    prepare_state, GroupInfoExtractor,
    DIM_DISTILBERT, DIM_DYNAMIC_RAW, DIM_ACTOR_INPUT,
)
from distilbert_encoder import DistilBERTEncoder, LMActor, LMCritic
from t1_run_multiseed import (
    get_device, set_global_seed, ReplayMemory, OUNoise
)


# =======================================================================
# [섹션 1] 조건 정의
# =======================================================================

CONDITION_A = "static_only"     # 동적변수 제거 (zeros)
CONDITION_B = "static_dynamic"  # 동적변수 포함 (현재 T1)

CONDITION_META = {
    CONDITION_A: {
        "label":   "Static-Only (768D)",
        "desc":    "DistilBERT 768D만 사용, 동적변수 제거",
        "use_dyn": False,
    },
    CONDITION_B: {
        "label":   "Static+Dynamic (771D)",
        "desc":    "DistilBERT 768D + 동적변수 3D 사용 (T1 기준)",
        "use_dyn": True,
    },
}


# =======================================================================
# [섹션 2] 조건별 LMDDPG 에이전트
# =======================================================================

class LMDDPG_Ablation:
    """
    동적변수 유/무를 제어할 수 있는 LMDDPG.

    use_dynamic=False → dynamic_numeric 자리를 zeros(3)으로 대체
    → 네트워크 구조(771D)는 동일하게 유지
    → 공정한 비교: 구조 차이 없이 입력 정보만 다름
    """
    def __init__(self, gamma, tau, hidden_size, action_space,
                 encoder, device, use_dynamic: bool = True):
        self.device      = device
        self.gamma       = gamma
        self.tau         = tau
        self.use_dynamic = use_dynamic  # ★ 핵심 제어 플래그

        self.actor        = LMActor(hidden_size, action_space, encoder).to(device)
        self.actor_target = LMActor(hidden_size, action_space, encoder).to(device)

        encoder_param_ids = {id(p) for p in encoder.parameters()}
        actor_fc_params   = [p for p in self.actor.parameters()
                             if id(p) not in encoder_param_ids]
        self.actor_optim  = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": 1e-5},
            {"params": actor_fc_params,       "lr": 3e-5},  # ★ [Frozen] 1e-4 → 3e-5
        ])

        self.critic        = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_target = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_optim  = torch.optim.Adam(self.critic.parameters(), lr=1e-4)  # ★ [Frozen] 1e-3 → 1e-4

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def _get_dyn(self, dyn_tensor: torch.Tensor) -> torch.Tensor:
        """
        use_dynamic=False이면 동적변수를 zeros로 대체.
        → 네트워크 입력 형태(shape)는 동일하게 유지
        → 공정 비교: 구조 동일, 정보만 제거
        """
        if self.use_dynamic:
            return dyn_tensor
        # zeros: 동적 정보를 완전히 제거
        return torch.zeros_like(dyn_tensor)

    def select_action(self, prompt, dynamic_numeric):
        self.actor.eval()
        dyn = self._get_dyn(dynamic_numeric)
        with torch.no_grad():
            mu = self.actor(prompt, dyn)
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

        dyn_t      = self._get_dyn(
            torch.FloatTensor(np.array(dynamics)).to(self.device))
        next_dyn_t = self._get_dyn(
            torch.FloatTensor(np.array(next_dynamics)).to(self.device))
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


Transition = namedtuple('Transition',
    ('state_91d', 'action', 'mask', 'next_state_91d', 'reward'))


# =======================================================================
# [섹션 3] 학습 루프
# =======================================================================

def train_one_seed(seed, n_episodes, device, group_extractor,
                   output_dir, use_dynamic: bool):
    os.makedirs(output_dir, exist_ok=True)
    cond_label = CONDITION_META[
        CONDITION_B if use_dynamic else CONDITION_A]["label"]

    logging.info(f"[{cond_label} | Seed {seed}] 학습 시작")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)

    # ★ [Frozen] trainable=False 고정 (FT 버전: trainable=not is_mac)
    encoder = DistilBERTEncoder(
        trainable=False, device=device, max_length=128)

    agent = LMDDPG_Ablation(
        gamma=0.7, tau=0.003, hidden_size=128,
        action_space=env.action_space,
        encoder=encoder, device=device,
        use_dynamic=use_dynamic,    # ★ 조건 분기
    )

    memory  = ReplayMemory(10000)
    ounoise = OUNoise(env.action_space.shape[0], sigma=0.1)  # ★ [Frozen] sigma 0.2→0.1

    # ★ [Frozen] Buffer Warmup 추가 — 초반 랜덤 경험으로 가중치 오염 방지
    WARMUP_STEPS = 32 * 4  # 128 step
    total_steps  = 0
    rewards, val_rewards, val_ctr = [], [], []

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

            next_91d, reward, done, _ = env.step(
                action_t.cpu().numpy().squeeze())

            memory.push(
                state_91d,
                action_t.detach().cpu().unsqueeze(0),
                torch.Tensor([not done]),
                next_91d,
                torch.Tensor([reward]),
            )
            state_91d = next_91d
            ep_reward += reward

            total_steps += 1
            if len(memory) > 32 and total_steps > WARMUP_STEPS:
                for _ in range(5):
                    batch = Transition(*zip(*memory.sample(32)))
                    agent.update_parameters(batch, group_extractor)

            if done:
                break

        rewards.append(ep_reward)

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
                    a = agent.select_action(p, d).cpu().numpy().squeeze()
                    ns, r, done, _ = env.step(a)
                    val_r += r; val_s += 1
                    if step_cnt == 0:
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
                f"  [Ep {i_ep:4d}] AvgReward: {avg_r:.4f} | "
                f"CTR: {ctr:.4f} | ColdCTR: {cold_ctr:.4f}"
            )

    np.save(f"{output_dir}/val_ctr.npy",     np.array(val_ctr))
    np.save(f"{output_dir}/val_rewards.npy", np.array(val_rewards))
    np.save(f"{output_dir}/rewards.npy",     np.array(rewards))
    agent.save_model(f"{output_dir}/model")
    env.close()
    logging.info(f"  [{cond_label} | Seed {seed}] 학습 완료")
    return {"val_ctr": val_ctr, "val_rewards": val_rewards}


# =======================================================================
# [섹션 4] 평가 루프
# =======================================================================

def evaluate_one_seed(seed, device, group_extractor, output_dir,
                      use_dynamic: bool, n_eval: int = 5000):
    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 9999, env)

    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)
    agent   = LMDDPG_Ablation(
        0.7, 0.003, 128, env.action_space, encoder, device,
        use_dynamic=use_dynamic,
    )
    agent.load_model(f"{output_dir}/model")
    agent.actor.eval()

    total_r, total_s, cold_r, cold_n = 0.0, 0, 0.0, 0

    for _ in range(n_eval):
        s = env.reset()
        step_cnt = 0
        while True:
            res = prepare_state(s, group_extractor)
            p   = res['prompt']
            d   = torch.FloatTensor(
                res['dynamic_numeric']).unsqueeze(0).to(device)
            a   = agent.select_action(p, d).cpu().numpy().squeeze()
            ns, r, done, _ = env.step(a)
            if step_cnt == 0:
                cold_r += r; cold_n += 1
            total_r += r; total_s += 1
            step_cnt += 1
            s = ns
            if done: break

    cond_label = CONDITION_META[
        CONDITION_B if use_dynamic else CONDITION_A]["label"]
    result = {
        "condition":    cond_label,
        "use_dynamic":  use_dynamic,
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
        f"  [{cond_label} | Seed {seed}] "
        f"CTR: {result['CTR']:.4f} | ColdCTR: {result['ColdStartCTR']:.4f}"
    )
    env.close()
    return result


# =======================================================================
# [섹션 5] 결과 집계 및 시각화
# =======================================================================

def save_condition_summary(results: list, cond_dir: str, use_dynamic: bool):
    """단일 조건 내 여러 seed 결과 요약."""
    path = f"{cond_dir}/summary.csv"
    fields = ["condition", "use_dynamic", "seed",
              "AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
        for m in ["AvgReward", "CTR", "ColdStartCTR"]:
            vals = [r[m] for r in results]
            writer.writerow({
                "condition": "mean±std", "use_dynamic": use_dynamic,
                "seed": "─",
                m: f"{np.mean(vals):.4f}±{np.std(vals):.4f}",
            })


def save_ablation_comparison(results_a: list, results_b: list,
                              output_dir: str):
    """
    A vs B 비교표 저장 + 콘솔 출력.
    논문 Ablation Study Table 직접 활용 가능.
    """
    path = f"{output_dir}/ablation_comparison.csv"
    rows = []
    for cond_label, results in [
        ("Static-Only (768D)", results_a),
        ("Static+Dynamic (771D)", results_b),
    ]:
        for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[m] for r in results]
            rows.append({
                "condition": cond_label,
                "metric":    m,
                "mean":      round(np.mean(vals), 4),
                "std":       round(np.std(vals),  4),
                "min":       round(min(vals),      4),
                "max":       round(max(vals),      4),
                "n_seeds":   len(results),
            })

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # 콘솔 출력
    print(f"\n{'='*65}")
    print("  동적변수 소거 실험 (Ablation) 결과")
    print(f"{'='*65}")
    print(f"  {'조건':<25} {'AvgReward':>12} {'CTR':>10} {'ColdCTR':>12}")
    print(f"  {'─'*60}")
    for cond_label, results in [
        ("Static-Only (768D)",     results_a),
        ("Static+Dynamic (771D)",  results_b),
    ]:
        for m_idx, m in enumerate(["AvgReward", "CTR", "ColdStartCTR"]):
            vals = [r[m] for r in results]
            if m_idx == 0:
                print(f"  {cond_label:<25}", end="")
            else:
                print(f"  {'':25}", end="")
            print(f"  {m}: {np.mean(vals):.4f}±{np.std(vals):.4f}")

    # CTR 향상률 계산
    ctr_a = np.mean([r["CTR"] for r in results_a])
    ctr_b = np.mean([r["CTR"] for r in results_b])
    cold_a = np.mean([r["ColdStartCTR"] for r in results_a])
    cold_b = np.mean([r["ColdStartCTR"] for r in results_b])
    print(f"\n  CTR 향상률 (B-A)/A:     {(ctr_b-ctr_a)/ctr_a*100:+.2f}%")
    print(f"  ColdCTR 향상률 (B-A)/A: {(cold_b-cold_a)/cold_a*100:+.2f}%")

    if ctr_b > ctr_a:
        print(f"\n  ★ 동적변수 3D가 CTR 향상에 기여함 → 연구 주장 실험 증명")
    else:
        print(f"\n  ▲ 동적변수 제거 시에도 CTR 유사 → 추가 분석 필요")
    print(f"{'='*65}\n")
    logging.info(f"비교표 저장: {path}")


def plot_ablation_comparison(results_a: list, results_b: list,
                              seeds: list, output_dir: str):
    """A vs B 비교 시각화 저장."""
    metrics  = ["AvgReward", "CTR", "ColdStartCTR"]
    labels   = ["Static-Only\n(768D)", "Static+Dynamic\n(771D)"]
    colors   = ["#7CB9E8", "#1B4F8A"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Dynamic Variable Ablation Study\n(T1 Condition, mean ± std)",
                 fontsize=13)

    for ax, metric in zip(axes, metrics):
        vals_a = [r[metric] for r in results_a]
        vals_b = [r[metric] for r in results_b]
        means  = [np.mean(vals_a), np.mean(vals_b)]
        stds   = [np.std(vals_a),  np.std(vals_b)]

        bars = ax.bar(labels, means, color=colors, alpha=0.85,
                      yerr=stds, capsize=6, error_kw={"linewidth": 2})

        # 수치 표시
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + 0.005,
                    f"{mean:.3f}\n±{std:.3f}",
                    ha="center", va="bottom", fontsize=8)

        # 개별 seed 점 표시 (분포 확인)
        for j, (vals, x_base) in enumerate([(vals_a, 0), (vals_b, 1)]):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(
                [x_base + j_ for j_ in jitter], vals,
                color="black", s=15, alpha=0.4, zorder=5
            )

        ax.set_title(metric, fontsize=11)
        ax.set_ylabel("Score", fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ablation_comparison.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    logging.info(f"시각화 저장: {output_dir}/ablation_comparison.png")

    # 학습 곡선 비교 (CTR)
    _plot_ctr_curves(results_a, results_b, seeds, output_dir)


def _plot_ctr_curves(results_a, results_b, seeds, output_dir):
    """두 조건의 CTR 학습 곡선 비교."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CTR Learning Curves: Static-Only vs Static+Dynamic",
                 fontsize=12)

    colors = plt.cm.tab10(np.linspace(0, 1, len(seeds)))

    for ax, (cond_dir, title, results) in zip(axes, [
        (CONDITION_A, "Static-Only (768D)", results_a),
        (CONDITION_B, "Static+Dynamic (771D)", results_b),
    ]):
        ctr_curves = []
        for seed in seeds:
            p = f"{output_dir}/{cond_dir}/seed_{seed}/val_ctr.npy"
            if os.path.exists(p):
                arr = np.load(p)
                ctr_curves.append(arr)
                x = np.arange(len(arr)) * 20
                ax.plot(x, arr, color=colors[seeds.index(seed)],
                        alpha=0.3, linewidth=1)

        if ctr_curves:
            min_len = min(len(c) for c in ctr_curves)
            arr = np.array([c[:min_len] for c in ctr_curves])
            x   = np.arange(min_len) * 20
            ax.plot(x, arr.mean(0), color='black', linewidth=2.5,
                    label='Mean')
            ax.fill_between(x, arr.mean(0) - arr.std(0),
                            arr.mean(0) + arr.std(0),
                            color='gray', alpha=0.25, label='±1 Std')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("CTR", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ctr_curves_comparison.png", dpi=150)
    plt.close()
    logging.info(f"학습 곡선 저장: {output_dir}/ctr_curves_comparison.png")


# =======================================================================
# [섹션 6] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="동적변수 소거 실험 (Ablation)")
    parser.add_argument("--seeds",      nargs="+", type=int,
                        default=list(range(10)),
                        help="실험 seed (기본: 0~9)")
    parser.add_argument("--n_episodes", type=int, default=2000)
    parser.add_argument("--n_eval",     type=int, default=5000)
    parser.add_argument("--eval_only",  action="store_true")
    parser.add_argument("--gpu",        type=int, default=None)
    parser.add_argument("--output_dir", default="output/exp_dynamic_ablation_frozen")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{args.output_dir}/exp_dynamic_ablation_frozen.log"),
        ],
    )

    device = get_device()
    logging.info("="*55)
    logging.info("동적변수 유효성 검증 실험 (Ablation Study, T1 Frozen 기반) 시작")
    logging.info(f"  조건 A: Static-Only  (768D — 동적변수 제거)")
    logging.info(f"  조건 B: Static+Dyn   (771D — 동적변수 포함, T1 기준)")
    logging.info(f"  Seeds: {args.seeds}  ({len(args.seeds)}개)")
    logging.info(f"  디바이스: {device}")
    logging.info("="*55)

    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    results_a, results_b = [], []

    for use_dyn, cond_key, result_list in [
        (False, CONDITION_A, results_a),
        (True,  CONDITION_B, results_b),
    ]:
        cond_dir  = f"{args.output_dir}/{cond_key}"
        cond_meta = CONDITION_META[cond_key]
        logging.info(f"\n[{cond_meta['label']}] 실험 시작")

        for seed in args.seeds:
            seed_dir = f"{cond_dir}/seed_{seed}"

            if not args.eval_only:
                train_one_seed(
                    seed=seed, n_episodes=args.n_episodes,
                    device=device, group_extractor=group_extractor,
                    output_dir=seed_dir, use_dynamic=use_dyn,
                )

            model_path = f"{seed_dir}/model_actor.pt"
            if os.path.exists(model_path):
                result = evaluate_one_seed(
                    seed=seed, device=device,
                    group_extractor=group_extractor,
                    output_dir=seed_dir,
                    use_dynamic=use_dyn,
                    n_eval=args.n_eval,
                )
                result_list.append(result)

        if result_list:
            save_condition_summary(result_list, cond_dir, use_dyn)

    # ── 비교 결과 집계 + 시각화 ───────────────────────────────────────
    if results_a and results_b:
        save_ablation_comparison(results_a, results_b, args.output_dir)
        plot_ablation_comparison(results_a, results_b, args.seeds,
                                 args.output_dir)

    logging.info(f"T1 Frozen Ablation 실험 완료. 결과: {args.output_dir}/")


if __name__ == "__main__":
    main()