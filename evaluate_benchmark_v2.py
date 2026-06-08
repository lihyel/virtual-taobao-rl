"""
=======================================================================
evaluate_benchmark.py  —  전체 벤치마크 평가 스크립트 (수정판)
=======================================================================

[v1 → v2 수정 내역]

  Q1. T0 baseline의 91D 수치벡터 처리 방식을 김홍과 일치시킴

  수정 1: _NumericalActor 내부 LayerNorm 구현 교체
    Before (v1): x.mean(-1, keepdim=True)  ← 마지막 차원 정규화
    After  (v2): x.view(x.size(0),-1).mean(1) 방식  ← 김홍 ddpg.py와 동일
    이유: exp3_t0_baseline_v2.py도 view 방식으로 수정됨
         학습(exp3_v2) ↔ 평가(benchmark)의 LayerNorm이 반드시 일치해야 함

  수정 2: benchmark_t0_numerical의 state 처리를 김홍 evaluate.py와 동일하게
    Before: torch.FloatTensor(state).unsqueeze(0)
    After:  torch.Tensor([state])  ← 김홍 infer_mdp_p()와 완전 동일
    결과: 두 방식 모두 (1,91) shape이라 기능상 동일하지만 일관성 확보

  수정 3: action 추출 방식을 김홍 evaluate.py와 동일하게
    Before: actor(s).clamp(-1,1).cpu().numpy().squeeze()
    After:  actor(s).cpu().numpy()[0]  ← 김홍 방식 그대로

  Q2. AvgReward_KimHong 계산 방식 각주 명시 추가
    → 김홍 논문 수식과 맥락이 다름을 주석으로 명확히 표기

[파일 위치]
  your_project/evaluate_benchmark.py

[실행]
  python evaluate_benchmark.py                          # 기본 3000 에피소드
  python evaluate_benchmark.py --n_episodes 5000        # 5000 에피소드
  python evaluate_benchmark.py --t0_model models/exp3_t0_actor_final
=======================================================================
"""

import os
import csv
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import gym
import virtualTB

from t1_textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder, LMActor
from t1_trainer_lm_ddpg import get_device


# =======================================================================
# [섹션 1] 공통 환경 설정
# =======================================================================

def make_env(seed: int = 42) -> gym.Env:
    """
    모든 모델이 동일 seed로 평가받아야 공정한 비교가 됨.
    동일 seed → 동일한 사용자 시퀀스 생성 → 공정한 비교.
    """
    env = gym.make('VirtualTB-v0')
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env


def run_episodes(env, action_fn, n_episodes: int, desc: str,
                 group_size: int = 10) -> dict:
    """
    공통 평가 루프. 모든 모델이 이 함수를 통해 동일하게 평가됨.

    [AvgReward 두 가지 방식]
    방식1 (본 연구):    Σ클릭 / 에피소드 수  (단순 평균, 보수적)
    방식2 (김홍 참고):  에피소드를 group_size개씩 묶어 각 그룹 최대값 평균

    주의: 방식2는 김홍 논문의 "Average reward100 Approx."와 유사하나
          김홍은 학습 중 100번 검증값 기준이고, 여기서는 최종 평가 기준으로
          적용하므로 직접 수치 비교 시 주의 필요.
          → CTR을 주 비교 지표로 사용 권장.
    """
    total_reward    = 0.0
    total_steps     = 0
    cold_reward     = 0.0
    cold_count      = 0
    episode_rewards = []

    for _ in tqdm(range(n_episodes), desc=f"  {desc:<35}"):
        state_91d = env.reset()
        ep_reward = 0.0
        step      = 0

        while True:
            action = action_fn(state_91d)
            next_state, reward, done, _ = env.step(action)

            if step == 0:
                cold_reward += reward
                cold_count  += 1

            ep_reward   += reward
            total_steps += 1
            step        += 1
            state_91d    = next_state

            if done:
                break

        total_reward += ep_reward
        episode_rewards.append(ep_reward)

    avg_reward = total_reward / n_episodes

    groups = [
        episode_rewards[i:i + group_size]
        for i in range(0, len(episode_rewards), group_size)
        if episode_rewards[i:i + group_size]
    ]
    avg_reward_kim = float(np.mean([max(g) for g in groups]))

    ctr       = total_reward / total_steps / 10 if total_steps > 0 else 0
    avg_steps = total_steps  / n_episodes
    cold_ctr  = cold_reward  / cold_count / 10 if cold_count > 0 else 0

    return {
        "AvgReward":         round(avg_reward, 4),
        "AvgReward_KimHong": round(avg_reward_kim, 4),
        "CTR":               round(ctr, 4),
        "AvgSteps":          round(avg_steps, 2),
        "ColdStartCTR":      round(cold_ctr, 4),
        "n_episodes":        n_episodes,
        "group_size":        group_size,
    }


# =======================================================================
# [섹션 2] 계층1 — 최하 기준선
# =======================================================================

def benchmark_random(env, n_episodes: int) -> dict:
    """무작위 행동. 김홍 eval_random()과 동일."""
    def action_fn(state_91d):
        return (np.random.rand(27) - 0.5) * 2

    result = run_episodes(env, action_fn, n_episodes, "Random Policy")
    result["model"] = "Random Policy"
    result["tier"]  = "계층1 기준선"
    return result


def benchmark_popularity(env, n_episodes: int) -> dict:
    """고정 최대 가중치 Rule-based 정책."""
    fixed_action = np.ones(27, dtype=np.float32)

    def action_fn(state_91d):
        return fixed_action

    result = run_episodes(env, action_fn, n_episodes,
                          "Popularity Heuristic (Rule-based)")
    result["model"] = "Popularity Heuristic"
    result["tier"]  = "계층1 기준선"
    return result


# =======================================================================
# [섹션 3] 계층2 — T0 수치 DDPG 베이스라인
# =======================================================================

class _LayerNorm(nn.Module):
    """
    김홍 ddpg.py의 LayerNorm과 완전히 동일한 구현.

    [수정 핵심]
    v1: x.mean(-1, keepdim=True)  → 마지막 차원 기준
    v2: x.view(x.size(0),-1).mean(1)  → 김홍 ddpg.py와 동일한 batch 기준

    exp3_t0_baseline_v2.py의 학습 시 LayerNorm과 반드시 일치해야
    state_dict 로드 후 올바른 추론이 가능함.
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.affine       = affine
        self.eps          = eps
        if self.affine:
            # 파라미터 이름을 김홍 ddpg.py와 동일하게 유지
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta  = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # 김홍 ddpg.py와 완전히 동일한 연산
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std  = x.view(x.size(0), -1).std(1).view(*shape)
        y    = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class _NumericalActor(nn.Module):
    """
    T0 수치 DDPG Actor.
    exp3_t0_baseline_v2.py의 Actor와 레이어 이름까지 완전히 동일.

    [에러 수정]
    이전: self.linear1, self.linear2  → state_dict 키: "linear1.weight" 등
    수정: self.l1, self.l2            → state_dict 키: "l1.weight" 등
    이유: exp3_t0_baseline_v2.py에서 self.l1, self.l2로 학습했으므로
          평가 코드의 레이어 이름이 반드시 일치해야 load_state_dict 성공

    입력: 91D 수치 벡터 (torch.Tensor([state]) → shape (1,91))
    출력: 27D action
    """
    def __init__(self, hidden_size=128, num_inputs=91, action_space=None):
        super().__init__()
        num_outputs = action_space.shape[0] if action_space else 27

        # ★ exp3_t0_baseline_v2.py와 동일한 레이어 이름: l1, l2
        self.l1  = nn.Linear(num_inputs, hidden_size)
        self.ln1 = _LayerNorm(hidden_size)
        self.l2  = nn.Linear(hidden_size, hidden_size)
        self.ln2 = _LayerNorm(hidden_size)
        self.mu  = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return torch.tanh(self.mu(x))


def benchmark_t0_numerical(actor_path: str, env, device: str,
                            n_episodes: int) -> dict:
    """
    T0 수치 DDPG 평가 (exp3_t0_baseline_v2.py 학습 결과).

    [Q1 핵심]
    state 처리 방식을 김홍 evaluate.py의 infer_mdp_p()와 완전히 동일하게:
      김홍: state = torch.Tensor([state]).to(device)  → shape (1, 91)
            action = model.select_action(state).cpu()
            env.step(action.numpy()[0])               → [0]으로 배치 추출

      본 코드: 동일한 방식 적용
    """
    actor = _NumericalActor(
        hidden_size=128,
        num_inputs=env.observation_space.shape[0],   # 91
        action_space=env.action_space,
    ).to(device)
    # weights_only=False: LayerNorm 커스텀 파라미터 포함 로드 (FutureWarning 명시 해결)
    actor.load_state_dict(
        torch.load(actor_path, map_location=device, weights_only=False)
    )
    actor.eval()
    logging.info(f"T0 Actor 로드: {actor_path}")

    def action_fn(state_91d):
        # 김홍 infer_mdp_p()와 완전히 동일한 방식
        state = torch.Tensor([state_91d]).to(device)   # (1, 91)
        with torch.no_grad():
            action = actor(state).cpu()
        return action.numpy()[0]                        # (27,)

    result = run_episodes(env, action_fn, n_episodes,
                          "T0 Numerical DDPG (베이스라인)")
    result["model"] = "T0 수치 DDPG"
    result["tier"]  = "계층2 RL 기준선"
    return result


# =======================================================================
# [섹션 4] 계층3 — LM 기반 본 연구
# =======================================================================

def _make_lm_action_fn(actor_path: str, device: str, action_space,
                        encoder: DistilBERTEncoder,
                        group_extractor: GroupInfoExtractor,
                        from_checkpoint: bool = False):
    """
    LM-DDPG용 greedy action 함수 생성.
    noise 없이 순수 정책으로만 평가.
    """
    actor = LMActor(
        hidden_size=128,
        action_space=action_space,
        bert_encoder=encoder,
    ).to(device)

    if from_checkpoint:
        ckpt = torch.load(actor_path, map_location=device, weights_only=False)
        actor.load_state_dict(ckpt["actor_state"])
        logging.info(f"체크포인트 로드: {actor_path} (Ep {ckpt['episode']})")
    else:
        actor.load_state_dict(
            torch.load(actor_path, map_location=device, weights_only=False)
        )
        logging.info(f"LM Actor 로드: {actor_path}")

    actor.eval()

    def action_fn(state_91d):
        res    = prepare_state(state_91d, group_extractor)
        prompt = res["prompt"]
        dyn    = torch.FloatTensor(
            res["dynamic_numeric"]).unsqueeze(0).to(device)
        with torch.no_grad():
            action = actor([prompt], dyn).squeeze(0)
        return action.clamp(-1, 1).cpu().numpy()

    return action_fn


def benchmark_lm_frozen(actor_path: str, env, device: str,
                         encoder: DistilBERTEncoder,
                         group_extractor: GroupInfoExtractor,
                         n_episodes: int,
                         from_checkpoint: bool = False) -> dict:
    """T1 LM-DDPG Frozen: DistilBERT 고정, FC만 학습."""
    action_fn = _make_lm_action_fn(
        actor_path, device, env.action_space,
        encoder, group_extractor, from_checkpoint)
    result = run_episodes(env, action_fn, n_episodes, "T1 LM-DDPG Frozen")
    result["model"] = "T1 LM-DDPG Frozen"
    result["tier"]  = "계층3 본 연구"
    return result


def benchmark_lm_finetuned(actor_path: str, env, device: str,
                            encoder: DistilBERTEncoder,
                            group_extractor: GroupInfoExtractor,
                            n_episodes: int,
                            from_checkpoint: bool = False) -> dict:
    """T1 LM-DDPG Fine-tuning: DistilBERT + FC 함께 학습 (주 실험)."""
    action_fn = _make_lm_action_fn(
        actor_path, device, env.action_space,
        encoder, group_extractor, from_checkpoint)
    result = run_episodes(env, action_fn, n_episodes,
                          "T1 LM-DDPG Fine-tuning (본 연구)")
    result["model"] = "T1 LM-DDPG Fine-tuning"
    result["tier"]  = "계층3 본 연구"
    return result


# =======================================================================
# [섹션 5] 결과 출력 및 저장
# =======================================================================

def print_benchmark_table(results: list):
    """논문 결과표 형식으로 콘솔 출력."""
    print()
    print("=" * 105)
    print("  벤치마크 결과 (Virtual-Taobao)")
    print("=" * 105)
    header = (f"  {'모델':<35} "
              f"{'AvgRew(본연구)':>15} "
              f"{'AvgRew(참고)':>13} "
              f"{'CTR':>8} "
              f"{'AvgSteps':>10} "
              f"{'ColdCTR':>10}")
    print(header)
    print("-" * 105)

    current_tier = None
    for r in results:
        if r.get("tier") != current_tier:
            current_tier = r.get("tier", "")
            print(f"\n  [{current_tier}]")

        print(f"  {r['model']:<35} "
              f"{r['AvgReward']:>15.4f} "
              f"{r['AvgReward_KimHong']:>13.4f} "
              f"{r['CTR']:>8.4f} "
              f"{r['AvgSteps']:>10.2f} "
              f"{r['ColdStartCTR']:>10.4f}")

    print()
    print("  [참고: 김홍(2023) 논문 수치]")
    print(f"  {'MDP(P) 단독':<35} {'40.93':>28} {'0.541':>8}")
    print(f"  {'MDP(P)+G+MLP SOTA':<35} {'56.54':>28} {'0.767':>8}")
    print("=" * 105)
    print()
    print("  지표 설명:")
    print("  AvgRew(본연구) : Σ클릭 / 에피소드 수  — 단순 산술 평균 (★ 주 비교 지표)")
    print("  AvgRew(참고)   : 에피소드를 10개씩 묶어 각 그룹 최대값 평균 (참고용)")
    print("                   ※ 김홍 논문 방식과 맥락이 달라 직접 비교 부적합")
    print("  CTR            : Σ클릭 / Σstep / 10  — 김홍(2023)과 완전 동일 ✅ (★ 주 비교 지표)")
    print("  AvgSteps       : 평균 세션 길이 (본 연구 추가)")
    print("  ColdCTR        : 세션 첫 추천 CTR — Cold-start 핵심 지표 (본 연구 추가)")
    print()
    print("  ★ 김홍 논문 수치와의 비교: CTR 기준으로 비교하는 것이 가장 정확")


def save_benchmark_csv(results: list,
                        path: str = "output/benchmark_results.csv"):
    """결과를 CSV로 저장."""
    os.makedirs("output", exist_ok=True)
    keys = ["tier", "model",
            "AvgReward", "AvgReward_KimHong",
            "CTR", "AvgSteps", "ColdStartCTR",
            "n_episodes", "group_size"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logging.info(f"결과 저장: {path}")
    print(f"\n  → CSV 저장: {path}")


# =======================================================================
# [섹션 6] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="LM-DDPG 전체 벤치마크")
    parser.add_argument("--n_episodes", type=int, default=5000,
                        help="평가 에피소드 수 (기본 5000, 김홍과 동일)")
    parser.add_argument("--lm_finetuned",
                        default="models/lm_actor_LM_DDPG_T1_final",
                        help="LM Fine-tuning Actor 경로")
    parser.add_argument("--lm_frozen",
                        default="models/lm_actor_LM_DDPG_T1_Frozen",
                        help="LM Frozen Actor 경로")
    parser.add_argument("--t0_model",
                        default="models/exp3_t0_actor_final",
                        help="T0 수치 DDPG Actor 경로")
    parser.add_argument("--from_checkpoint", action="store_true",
                        help="모델이 checkpoint_epN.pt 형식일 때")
    parser.add_argument("--seed", type=int, default=42,
                        help="평가 환경 seed")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("benchmark.log"),
        ],
    )

    device = get_device()
    env    = make_env(seed=args.seed)
    N      = args.n_episodes

    logging.info(f"벤치마크 시작 | 에피소드: {N} | 디바이스: {device}")
    logging.info(f"T0 LayerNorm: view 방식 (김홍 ddpg.py 동일) ✅")
    logging.info(f"T0 state 처리: torch.Tensor([state]) 방식 (김홍 evaluate.py 동일) ✅")
    print(f"\n  [벤치마크] {N}에피소드 × 각 모델  |  device: {device}\n")

    # ── 공유 컴포넌트 ─────────────────────────────────────────────
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    encoder = DistilBERTEncoder(
        trainable=False, device=device, max_length=128
    )

    results = []

    # ── 계층1: 기준선 ─────────────────────────────────────────────
    logging.info("계층1: 기준선 평가")
    results.append(benchmark_random(env, N))
    results.append(benchmark_popularity(env, N))

    # ── 계층2: T0 수치 DDPG ───────────────────────────────────────
    logging.info("계층2: T0 수치 DDPG 평가")
    if os.path.exists(args.t0_model):
        results.append(benchmark_t0_numerical(args.t0_model, env, device, N))
    else:
        logging.warning(f"T0 모델 없음: {args.t0_model} — 건너뜀")
        print(f"  ⚠ T0 모델 없음: {args.t0_model}")

    # ── 계층3: 본 연구 ────────────────────────────────────────────
    logging.info("계층3: LM 기반 모델 평가")
    if os.path.exists(args.lm_frozen):
        results.append(benchmark_lm_frozen(
            args.lm_frozen, env, device,
            encoder, group_extractor, N, args.from_checkpoint))
    else:
        logging.warning(f"LM Frozen 모델 없음: {args.lm_frozen}")
        print(f"  ⚠ LM Frozen 모델 없음: {args.lm_frozen}")

    if os.path.exists(args.lm_finetuned):
        results.append(benchmark_lm_finetuned(
            args.lm_finetuned, env, device,
            encoder, group_extractor, N, args.from_checkpoint))
    else:
        logging.warning(f"LM FT 모델 없음: {args.lm_finetuned}")
        print(f"  ⚠ LM FT 모델 없음: {args.lm_finetuned}")

    # ── 출력 및 저장 ──────────────────────────────────────────────
    print_benchmark_table(results)
    save_benchmark_csv(results)
    env.close()
    logging.info("벤치마크 완료.")


if __name__ == "__main__":
    main()