"""
=======================================================================
T2/t2_run_multiseed.py  —  T2 속성 순차 추가 다중 Seed 실험
=======================================================================

[파일 위치]
  your_project/
  ├── t1_textify.py
  ├── t1_run_multiseed.py      ← T1 실험 (비교 기준)
  └── T2/
      ├── t2_textify.py
      └── t2_run_multiseed.py  ← 이 파일

[실험 구조]
  Step 0 (T1 기준): 확정 4개 속성
  Step 1: + is_occupied
  Step 2: + intentions
  Step 3: + geography
  Step 4: + profile_type
  Step 5: + user_group
  Step 6: + shops
  Step 7: + brands

  각 Step마다 다중 Seed(기본 3개)로 반복 학습 + 평가
  → Step별 CTR/ColdCTR 비교로 각 속성의 기여도 측정

[출력 파일 구조]
  output/T2/
  ├── step1_occupied/
  │   ├── seed_0/  eval_result.csv, val_ctr.npy, ...
  │   ├── seed_1/
  │   ├── seed_42/
  │   └── summary.csv          ← 이 Step의 mean ± std
  ├── step2_intentions/ ...
  ├── all_steps_comparison.csv ← 전체 Step 비교표
  └── step_comparison.png      ← Step별 CTR/ColdCTR 그래프

[실행 방법]
  # 전체 Step 1~7 순차 실행 (seed 0, 1, 42)
  python T2/t2_run_multiseed.py

  # 특정 Step만 실행
  python T2/t2_run_multiseed.py --steps 1 2 3

  # seed 지정
  python T2/t2_run_multiseed.py --steps 1 2 --seeds 0 1 2 42
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t1_textify import GroupInfoExtractor, get_dynamic_numeric
from T2.t2_textify import prepare_state_T2, T2_STEP_META
from distilbert_encoder import DistilBERTEncoder, LMActor, LMCritic


# =======================================================================
# [섹션 1] 공통 유틸리티 (t1_run_multiseed.py와 동일)
# =======================================================================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_global_seed(seed: int, env=None):
    """전역 seed 설정 — 네트워크 초기화, 탐색 노이즈, 환경 샘플링 모두 통일."""
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
# [섹션 2] LMDDPG 에이전트
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

    def update_parameters(self, batch, group_extractor, t2_step: int):
        """
        T2 핵심 변경점: prepare_state_T2()로 텍스트화 시 step 인자 전달.
        """
        prompts, dynamics, next_prompts, next_dynamics = [], [], [], []
        for s, ns in zip(batch.state_91d, batch.next_state_91d):
            r  = prepare_state_T2(s,  group_extractor, step=t2_step)
            nr = prepare_state_T2(ns, group_extractor, step=t2_step)
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
# [섹션 3] 단일 Seed 학습
# =======================================================================

def train_one_seed(seed, n_episodes, device, group_extractor,
                   output_dir, t2_step):
    """
    단일 seed로 T2-StepN 학습.
    t1_run_multiseed.py와 구조 동일, t2_step 인자만 추가.
    """
    os.makedirs(output_dir, exist_ok=True)
    meta = T2_STEP_META[t2_step]
    logging.info(f"[{meta['name']} | Seed {seed}] 학습 시작")

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)

    is_mac  = (device == "mps")
    encoder = DistilBERTEncoder(trainable=not is_mac, device=device, max_length=128)
    agent   = LMDDPG(0.7, 0.003, 128, env.action_space, encoder, device)

    memory  = ReplayMemory(10000)
    ounoise = OUNoise(env.action_space.shape[0])

    rewards, val_rewards, val_ctr = [], [], []

    for i_ep in range(n_episodes):
        state_91d = env.reset()
        ounoise.reset()
        ep_reward = 0

        while True:
            res    = prepare_state_T2(state_91d, group_extractor, step=t2_step)
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

            if len(memory) > 32:
                for _ in range(5):
                    batch = Transition(*zip(*memory.sample(32)))
                    agent.update_parameters(batch, group_extractor, t2_step)

            if done:
                break

        rewards.append(ep_reward)

        if i_ep > 0 and i_ep % 20 == 0:
            val_r, val_s, cold_r, cold_n = 0, 0, 0, 0
            for _ in range(50):
                s = env.reset()
                s_res = prepare_state_T2(s, group_extractor, step=t2_step)
                p = s_res['prompt']
                d = torch.FloatTensor(
                    s_res['dynamic_numeric']).unsqueeze(0).to(device)
                step_cnt = 0
                while True:
                    a  = agent.select_action(p, d).cpu().numpy().squeeze()
                    ns, r, done, _ = env.step(a)
                    val_r += r; val_s += 1
                    if step_cnt == 0:
                        cold_r += r; cold_n += 1
                    nr = prepare_state_T2(ns, group_extractor, step=t2_step)
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
    agent.save_model(f"{output_dir}/model")
    env.close()
    return {"val_ctr": val_ctr, "val_rewards": val_rewards}


# =======================================================================
# [섹션 4] 단일 Seed 최종 평가
# =======================================================================

def evaluate_one_seed(seed, device, group_extractor, output_dir,
                      t2_step, n_eval=5000):
    env = gym.make('VirtualTB-v0')
    set_global_seed(seed + 9999, env)

    is_mac  = (device == "mps")
    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)
    agent   = LMDDPG(0.7, 0.003, 128, env.action_space, encoder, device)
    agent.load_model(f"{output_dir}/model")
    agent.actor.eval()

    total_r, total_s, cold_r, cold_n = 0.0, 0, 0.0, 0

    for _ in range(n_eval):
        s = env.reset()
        step_cnt = 0
        while True:
            res = prepare_state_T2(s, group_extractor, step=t2_step)
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

    meta   = T2_STEP_META[t2_step]
    result = {
        "t2_step":     t2_step,
        "attr_added":  meta['attr'],
        "seed":        seed,
        "AvgReward":   round(total_r / n_eval, 4),
        "CTR":         round(total_r / total_s / 10, 4) if total_s > 0 else 0,
        "ColdStartCTR":round(cold_r  / cold_n  / 10, 4) if cold_n  > 0 else 0,
        "AvgSteps":    round(total_s / n_eval, 2),
        "n_eval":      n_eval,
    }

    with open(f"{output_dir}/eval_result.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    logging.info(
        f"[Step{t2_step} | Seed {seed}] "
        f"CTR: {result['CTR']:.4f} | ColdCTR: {result['ColdStartCTR']:.4f}"
    )
    env.close()
    return result


# =======================================================================
# [섹션 5] Step별 결과 집계
# =======================================================================

def save_step_summary(step_results: list, step_dir: str, t2_step: int):
    """단일 Step 내 여러 seed 결과 요약."""
    path = f"{step_dir}/summary.csv"
    fields = ["t2_step", "attr_added", "seed",
              "AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(step_results)

        # mean ± std
        for m in ["AvgReward", "CTR", "ColdStartCTR"]:
            vals = [r[m] for r in step_results]
            writer.writerow({
                "t2_step": t2_step,
                "attr_added": "mean±std",
                "seed": "─",
                m: f"{np.mean(vals):.4f}±{np.std(vals):.4f}",
            })


def save_all_steps_comparison(all_results: list, output_dir: str):
    """전체 Step의 mean CTR을 하나의 CSV로 요약."""
    path = f"{output_dir}/all_steps_comparison.csv"
    # Step별 평균 계산
    from collections import defaultdict
    by_step = defaultdict(list)
    for r in all_results:
        by_step[r['t2_step']].append(r)

    rows = []
    for step in sorted(by_step.keys()):
        rlist = by_step[step]
        meta  = T2_STEP_META[step]
        rows.append({
            "t2_step":        step,
            "attr_added":     meta['attr'],
            "mean_AvgReward": round(np.mean([r['AvgReward']   for r in rlist]), 4),
            "std_AvgReward":  round(np.std( [r['AvgReward']   for r in rlist]), 4),
            "mean_CTR":       round(np.mean([r['CTR']         for r in rlist]), 4),
            "std_CTR":        round(np.std( [r['CTR']         for r in rlist]), 4),
            "mean_ColdCTR":   round(np.mean([r['ColdStartCTR']for r in rlist]), 4),
            "std_ColdCTR":    round(np.std( [r['ColdStartCTR']for r in rlist]), 4),
            "n_seeds":        len(rlist),
        })

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"전체 Step 비교표 저장: {path}")

    # 콘솔 출력
    print(f"\n{'='*65}")
    print("  T2 전체 Step 비교 결과")
    print(f"{'='*65}")
    print(f"  {'Step':<6} {'추가 속성':<20} {'mean CTR':<12} {'mean ColdCTR':<14}")
    print(f"  {'─'*55}")
    for r in rows:
        print(f"  Step{r['t2_step']:<3} "
              f"{r['attr_added']:<20} "
              f"{r['mean_CTR']:.4f}±{r['std_CTR']:.4f}  "
              f"{r['mean_ColdCTR']:.4f}±{r['std_ColdCTR']:.4f}")
    print(f"{'='*65}\n")

    return rows


def plot_step_comparison(step_rows: list, output_dir: str):
    """Step별 CTR/ColdCTR 비교 그래프 저장."""
    steps   = [r['t2_step']     for r in step_rows]
    labels  = [f"S{r['t2_step']}\n{r['attr_added'][:8]}" for r in step_rows]
    ctrs    = [r['mean_CTR']    for r in step_rows]
    ctr_std = [r['std_CTR']     for r in step_rows]
    cold    = [r['mean_ColdCTR']for r in step_rows]
    cold_std= [r['std_ColdCTR'] for r in step_rows]

    x = np.arange(len(steps))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, ctrs,  w, yerr=ctr_std,  label='CTR',
           capsize=4, color='steelblue', alpha=0.8)
    ax.bar(x + w/2, cold,  w, yerr=cold_std, label='ColdStartCTR',
           capsize=4, color='coral',     alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("T2: CTR & ColdStartCTR by Attribute Step (mean ± std)",
                 fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/step_comparison.png", dpi=150)
    plt.close()
    logging.info(f"Step 비교 그래프 저장: {output_dir}/step_comparison.png")


# =======================================================================
# [섹션 6] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="T2 속성 순차 추가 다중 Seed 실험")
    # ── [추가] GPU 번호 지정을 위한 인자 ────────────────────────────────
    parser.add_argument('--gpu', type=int, default=1, help='사용할 GPU 번호 (0-7)')
    # ──────────────────────────────────────────────────────────────────
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(1, 8)),
                        help="실험할 T2 Step 목록 (기본: 1~7)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 42],
                        help="실험 seed 목록 (기본: 0 1 42)")
    parser.add_argument("--n_episodes", type=int, default=2000,
                        help="학습 에피소드 수")
    parser.add_argument("--n_eval",     type=int, default=5000,
                        help="평가 에피소드 수")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습 건너뛰고 평가만")
    parser.add_argument("--output_dir", default="output/T2",
                        help="결과 저장 루트 디렉토리")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/t2_multiseed.log"),
        ],
    )

    # ── [추가][핵심] 해당 프로세스가 지정된 GPU만 바라보도록 강제 설정 ──────────
    # 이 설정이 torch import 직후나 초기화 전에 실행되어야 가장 확실
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 이제 torch.device("cuda")를 호출하면 시스템의 n번 GPU가 
    # 이 프로세스 내부에서는 "cuda:0"으로 매핑됩니다.
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(f"선택된 GPU 번호: {args.gpu}")
    logging.info(f"현재 사용 중인 Device: {device}")
    # ──────────────────────────────────────────────────────────────────

    
    # K-Means는 모든 Step/Seed 공유
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    all_results = []

    # ── Step 순차 실행 ─────────────────────────────────────────────
    for t2_step in args.steps:
        meta     = T2_STEP_META[t2_step]
        step_dir = f"{args.output_dir}/{meta['folder']}"
        os.makedirs(step_dir, exist_ok=True)

        logging.info(f"\n{'='*55}")
        logging.info(f"[{meta['name']}] 속성: {meta['attr']}")
        logging.info(f"{'='*55}")

        step_results = []

        for seed in args.seeds:
            seed_dir = f"{step_dir}/seed_{seed}"

            if not args.eval_only:
                train_one_seed(
                    seed=seed, n_episodes=args.n_episodes,
                    device=device, group_extractor=group_extractor,
                    output_dir=seed_dir, t2_step=t2_step,
                )

            model_path = f"{seed_dir}/model_actor.pt"
            if os.path.exists(model_path):
                result = evaluate_one_seed(
                    seed=seed, device=device,
                    group_extractor=group_extractor,
                    output_dir=seed_dir,
                    t2_step=t2_step,
                    n_eval=args.n_eval,
                )
                step_results.append(result)
                all_results.append(result)
            else:
                logging.warning(f"  모델 없음: {model_path}")

        if step_results:
            save_step_summary(step_results, step_dir, t2_step)

    # ── 전체 Step 집계 + 시각화 ───────────────────────────────────
    if all_results:
        step_rows = save_all_steps_comparison(all_results, args.output_dir)
        plot_step_comparison(step_rows, args.output_dir)

    logging.info("T2 다중 Seed 실험 완료.")


if __name__ == "__main__":
    main()