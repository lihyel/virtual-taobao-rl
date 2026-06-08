"""
=======================================================================
benchmark.py  —  전체 벤치마크 평가 스크립트
=======================================================================

[파일 위치]
  your_project/benchmark.py

[역할]
  학습 완료된 모든 모델을 동일 조건(동일 환경, 동일 에피소드, 동일 seed)에서
  평가하여 논문 결과표를 생성하는 단일 진입점.

  교수님 피드백: "벤치마크 테스트 진행하라"
  → 내 모델 vs 기존 방법들을 같은 조건에서 수치로 비교

[평가 대상 (6종)]
  계층1 — 최하 기준선 (학습 불필요):
    1. Random Policy        : 무작위 27D 행동
    2. Popularity Heuristic : 모든 가중치=1 고정 (Rule-based)

  계층2 — RL 기준선:
    3. T0 수치 DDPG         : 91D 수치 직접 입력 (exp3_t0_baseline.py 학습 결과)
    4. MDP(P) 김홍(2023)    : 김홍 학습 모델 파일 있을 경우 (선택)

  계층3 — 본 연구 (Ablation):
    5. T1 LM-DDPG Frozen    : DistilBERT 고정 (exp2_t1_frozen.py 학습 결과)
    6. T1 LM-DDPG Fine-tune : DistilBERT 학습 (trainer_lm_ddpg_v2.py 학습 결과)

[평가 지표]
  AvgReward   : 에피소드 평균 누적 클릭 수 (김홍 동일)
  CTR         : Σ클릭 / Σstep / 10         (김홍 동일)
  AvgSteps    : 평균 세션 길이              (본 연구 추가)
  ColdStartCTR: Cold-start 첫 추천 CTR     (본 연구 추가)

[실행 방법]
  # 기본 실행 (3000 에피소드)
  python benchmark.py

  # 에피소드 수 지정
  python benchmark.py --n_episodes 5000

  # 김홍 MDP(P) 모델 포함
  python benchmark.py --mdp_p_path models/ddpg_actor_virtualTB_MDP(P)

[출력]
  콘솔: 논문 결과표 형식
  파일: output/benchmark_results.csv
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
from typing import Optional

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
    모든 모델이 동일한 환경 seed로 평가받아야 공정한 비교가 됨.
    seed를 고정하면 동일한 사용자 시퀀스가 생성됨.
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

    AvgReward를 두 가지 방식으로 계산합니다.

    [방식 1] AvgReward (본 연구, 보수적)
        = Σ(에피소드 누적 클릭 수) / n_episodes
        = 단순 산술 평균
        → 더 엄격한 기준. 이 수치에서 높으면 더 강한 주장.

    [방식 2] AvgReward_KimHong (김홍 방식, 낙관적)
        = mean( max(그룹 내 에피소드 보상) )
        근거: 김홍(2023) 논문 수식
              "Average reward100 Approx. = mean(round10 × max_reward10)"
              100번 검증을 10개씩 묶어 각 묶음의 최대값들의 평균
        → n_episodes개 에피소드를 group_size개씩 묶어 각 그룹 최대값 평균
        → 직접 비교를 위해 김홍과 동일 방식으로 계산

    Args:
        env        : VirtualTB 환경 (동일 seed)
        action_fn  : state_91d → action(27D) 함수
        n_episodes : 평가 에피소드 수 (group_size의 배수 권장)
        desc       : tqdm 표시용 모델명
        group_size : 김홍 방식에서 그룹 크기 (기본 10, 김홍과 동일)

    Returns:
        dict: AvgReward, AvgReward_KimHong, CTR, AvgSteps, ColdStartCTR
    """
    total_reward   = 0.0
    total_steps    = 0
    cold_reward    = 0.0
    cold_count     = 0
    episode_rewards = []   # 에피소드별 개별 보상 저장 (김홍 방식 계산용)

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
        episode_rewards.append(ep_reward)   # 에피소드 누적 보상 기록

    # ── 방식 1: 단순 산술 평균 (본 연구) ────────────────────────────
    avg_reward = total_reward / n_episodes

    # ── 방식 2: 그룹 최대값 평균 (김홍 방식) ─────────────────────────
    # episode_rewards를 group_size개씩 묶어 각 그룹의 최대값을 추출
    # 나머지 에피소드는 마지막 그룹에 포함
    # 예: 3000 에피소드, group_size=10 → 300그룹 → 300개 최대값 → 평균
    groups = [
        episode_rewards[i : i + group_size]
        for i in range(0, len(episode_rewards), group_size)
        if episode_rewards[i : i + group_size]  # 빈 그룹 제외
    ]
    avg_reward_kim = float(np.mean([max(g) for g in groups]))

    ctr       = total_reward / total_steps / 10 if total_steps > 0 else 0
    avg_steps = total_steps  / n_episodes
    cold_ctr  = cold_reward  / cold_count / 10 if cold_count > 0 else 0

    return {
        "AvgReward":        round(avg_reward, 4),       # 본 연구 방식
        "AvgReward_KimHong": round(avg_reward_kim, 4),  # 김홍 방식
        "CTR":              round(ctr, 4),
        "AvgSteps":         round(avg_steps, 2),
        "ColdStartCTR":     round(cold_ctr, 4),
        "n_episodes":       n_episodes,
        "group_size":       group_size,
    }


# =======================================================================
# [섹션 2] 계층1 — 최하 기준선 (학습 불필요)
# =======================================================================

def benchmark_random(env, n_episodes: int) -> dict:
    """
    무작위 행동 정책.
    김홍 evaluate.py의 eval_random()과 동일한 방식.
    27D 균등 분포 샘플링 ∈ [-1, 1]
    """
    def action_fn(state_91d):
        return (np.random.rand(27) - 0.5) * 2

    result = run_episodes(env, action_fn, n_episodes, "Random Policy")
    result["model"] = "Random Policy"
    result["tier"]  = "계층1 기준선"
    return result


def benchmark_popularity(env, n_episodes: int) -> dict:
    """
    인기도 기반 Rule-based 정책.
    모든 아이템 가중치를 최대값(1.0)으로 고정.
    '어떤 아이템이든 동등하게 선호' = 가장 단순한 비ML 기준선.

    이 정책보다 LM-DDPG가 높아야 '선택적 개인화'를 학습했다고 주장 가능.
    """
    fixed_action = np.ones(27, dtype=np.float32)

    def action_fn(state_91d):
        return fixed_action

    result = run_episodes(env, action_fn, n_episodes, "Popularity Heuristic (Rule-based)")
    result["model"] = "Popularity Heuristic"
    result["tier"]  = "계층1 기준선"
    return result


# =======================================================================
# [섹션 3] 계층2 — RL 기준선
# =======================================================================

class _NumericalActor(nn.Module):
    """
    T0 수치 기반 DDPG Actor.
    exp3_t0_baseline.py의 Actor와 완전히 동일한 구조.
    외부 파일 의존성 없이 인라인 정의.
    입력: 91D 수치 → 출력: 27D action
    """
    class _LN(nn.Module):
        def __init__(self, n):
            super().__init__()
            # 파일에 저장된 이름인 gamma와 beta로 변경합니다.
            self.gamma = nn.Parameter(torch.ones(n))
            self.beta = nn.Parameter(torch.zeros(n))
        def forward(self, x):
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True)
            # 수식에서도 변경된 변수명을 사용합니다.
            return self.gamma * (x - m) / (s + 1e-5) + self.beta

    def __init__(self, num_inputs=91, hidden=128, num_outputs=27):
        super().__init__()
        LN = _NumericalActor._LN
        self.l1 = nn.Linear(num_inputs, hidden); self.ln1 = LN(hidden)
        self.l2 = nn.Linear(hidden, hidden);     self.ln2 = LN(hidden)
        self.mu = nn.Linear(hidden, num_outputs)
        self.mu.weight.data.mul_(0.1); self.mu.bias.data.mul_(0.1)

    def forward(self, x):
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return torch.tanh(self.mu(x))


def benchmark_t0_numerical(actor_path: str, env, device: str,
                            n_episodes: int) -> dict:
    """
    T0: 수치 기반 DDPG (exp3_t0_baseline.py 학습 결과).
    LM 없이 91D 수치 직접 입력.
    LM-DDPG와의 차이 = LM 도입 자체의 효과.
    """
    actor = _NumericalActor().to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    def action_fn(state_91d):
        s = torch.FloatTensor(state_91d).unsqueeze(0).to(device)
        with torch.no_grad():
            return actor(s).clamp(-1, 1).cpu().numpy().squeeze()

    result = run_episodes(env, action_fn, n_episodes,
                          "T0 Numerical DDPG (김홍 재현)")
    result["model"] = "T0 수치 DDPG"
    result["tier"]  = "계층2 RL 기준선"
    return result


def benchmark_kim_mdp_p(actor_path: str, env, device: str,
                         n_episodes: int) -> dict:
    """
    김홍(2023) MDP(P) 모델 평가.
    김홍의 ddpg_cuda.py 구조와 동일 (91D 수치 입력).
    김홍이 직접 학습한 모델 파일이 있을 때 사용.

    actor_path 예시: models/ddpg_actor_virtualTB_MDP(P)
    """
    # 구조가 _NumericalActor와 동일 (91D → 128 → 128 → 27)
    actor = _NumericalActor(num_inputs=91, hidden=128, num_outputs=27).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    def action_fn(state_91d):
        s = torch.FloatTensor(state_91d).unsqueeze(0).to(device)
        with torch.no_grad():
            return actor(s).clamp(-1, 1).cpu().numpy().squeeze()

    result = run_episodes(env, action_fn, n_episodes,
                          "MDP(P) 김홍(2023)")
    result["model"] = "MDP(P) 김홍(2023)"
    result["tier"]  = "계층2 RL 기준선"
    return result


# =======================================================================
# [섹션 4] 계층3 — 본 연구 (LM 기반)
# =======================================================================

def _make_lm_action_fn(actor_path: str, device: str, action_space,
                        encoder: DistilBERTEncoder,
                        group_extractor: GroupInfoExtractor,
                        from_checkpoint: bool = False):
    """
    LM-DDPG용 action 함수 팩토리.
    Greedy action (noise 없음) — 순수 정책으로만 평가.

    from_checkpoint=True: checkpoint_epN.pt 파일에서 로드
    from_checkpoint=False: save_model()이 저장한 state_dict 파일에서 로드
    """
    actor = LMActor(
        hidden_size=128,
        action_space=action_space,
        bert_encoder=encoder,
    ).to(device)

    if from_checkpoint:
        ckpt = torch.load(actor_path, map_location=device)
        actor.load_state_dict(ckpt["actor_state"])
        logging.info(f"체크포인트 로드: {actor_path} (Ep {ckpt['episode']})")
    else:
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        logging.info(f"Actor 로드: {actor_path}")

    actor.eval()

    def action_fn(state_91d):
        res    = prepare_state(state_91d, group_extractor)
        prompt = res["prompt"]
        dyn    = torch.FloatTensor(res["dynamic_numeric"]).unsqueeze(0).to(device)
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
        encoder, group_extractor, from_checkpoint
    )
    result = run_episodes(env, action_fn, n_episodes,
                          "T1 LM-DDPG Frozen")
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
        encoder, group_extractor, from_checkpoint
    )
    result = run_episodes(env, action_fn, n_episodes,
                          "T1 LM-DDPG Fine-tuning (본 연구)")
    result["model"] = "T1 LM-DDPG Fine-tuning"
    result["tier"]  = "계층3 본 연구"
    return result


# =======================================================================
# [섹션 5] 결과 출력 및 저장
# =======================================================================

def print_benchmark_table(results: list):
    """
    논문 결과표 형식으로 콘솔 출력.
    AvgReward를 두 가지 방식으로 모두 출력하여 직접 비교 가능.
    """
    print()
    print("=" * 100)
    print("  벤치마크 결과 (Virtual-Taobao)")
    print("=" * 100)
    header = (f"  {'모델':<35} "
              f"{'AvgRew(본연구)':>15} "
              f"{'AvgRew(김홍방식)':>17} "
              f"{'CTR':>8} "
              f"{'AvgSteps':>10} "
              f"{'ColdCTR':>10}")
    print(header)
    print("-" * 100)

    # 계층별 구분선 출력
    current_tier = None
    for r in results:
        if r.get("tier") != current_tier:
            current_tier = r.get("tier", "")
            print(f"\n  [{current_tier}]")

        print(f"  {r['model']:<35} "
              f"{r['AvgReward']:>15.4f} "
              f"{r['AvgReward_KimHong']:>17.4f} "
              f"{r['CTR']:>8.4f} "
              f"{r['AvgSteps']:>10.2f} "
              f"{r['ColdStartCTR']:>10.4f}")

    print()
    print("  [참고: 김홍(2023) 논문 수치 (김홍 방식 AvgReward, CTR)]")
    print(f"  {'MDP(P) 단독':<35} {'40.93':>32} {'0.541':>8}")
    print(f"  {'MDP(P)+G+MLP SOTA':<35} {'56.54':>32} {'0.767':>8}")
    print("=" * 100)
    print()
    print("  지표 설명:")
    print("  AvgRew(본연구)   : Σ클릭 / 에피소드 수 — 단순 산술 평균 (보수적)")
    print("  AvgRew(김홍방식) : mean(max per group) — 그룹 최대값 평균 (낙관적)")
    print(f"                     김홍(2023) 수식: mean(round10 × max_reward10)")
    print(f"                     본 구현: {results[0].get('group_size', 10)}개 에피소드 그룹의 최대값들의 평균")
    print("  CTR              : Σ클릭 / Σstep / 10 — 김홍(2023)과 완전 동일 ✅")
    print("  AvgSteps         : 평균 세션 길이 (본 연구 추가 지표)")
    print("  ColdCTR          : 세션 첫 추천 CTR (본 연구 Cold-start 핵심 지표)")
    print()
    print("  ★ 주 비교 지표: CTR (두 연구 완전 동일)")
    print("  ★ AvgReward 비교 시: 김홍 방식끼리 비교 (AvgRew_KimHong vs 김홍 논문 수치)")


def save_benchmark_csv(results: list,
                        path: str = "output/benchmark_results.csv"):
    """결과를 CSV로 저장 — 논문 표에 바로 활용."""
    os.makedirs("output", exist_ok=True)
    keys = ["tier", "model",
            "AvgReward", "AvgReward_KimHong",   # 두 방식 모두 저장
            "CTR", "AvgSteps", "ColdStartCTR",
            "n_episodes", "group_size"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logging.info(f"결과 저장 완료: {path}")
    print(f"\n  → CSV 저장: {path}")


# =======================================================================
# [섹션 6] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="LM-DDPG 전체 벤치마크")

    # 에피소드 수
    parser.add_argument("--n_episodes", type=int, default=3000,
                        help="평가 에피소드 수 (기본 3000)")

    # 본 연구 모델 경로
    parser.add_argument("--lm_finetuned",
                        default="models/lm_actor_LM_DDPG_T1_final",
                        help="LM Fine-tuning Actor 경로")
    parser.add_argument("--lm_frozen",
                        default="models/lm_actor_LM_DDPG_T1_Frozen",
                        help="LM Frozen Actor 경로")

    # 수치 DDPG 경로
    parser.add_argument("--t0_model",
                        default="models/exp3_t0_actor_final",
                        help="T0 수치 DDPG Actor 경로")

    # 김홍 MDP(P) 경로 (선택)
    parser.add_argument("--mdp_p_path", default=None,
                        help="김홍(2023) MDP(P) Actor 경로 (없으면 건너뜀)")

    # 체크포인트 여부
    parser.add_argument("--from_checkpoint", action="store_true",
                        help="모델이 checkpoint_epN.pt 형식일 때")

    parser.add_argument("--seed", type=int, default=42)
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
    print(f"\n  [벤치마크] {N}에피소드 × 각 모델  |  device: {device}\n")

    # ── 공유 컴포넌트 (LM 모델들이 공통으로 사용) ────────────────────
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()  # 이미 저장된 K-Means 로드

    # DistilBERT: 평가 시에는 항상 frozen (추론 속도 최적화)
    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)

    results = []

    # ── 계층1: 최하 기준선 ─────────────────────────────────────────
    logging.info("계층1: 기준선 평가")
    results.append(benchmark_random(env, N))
    results.append(benchmark_popularity(env, N))

    # ── 계층2: RL 기준선 ──────────────────────────────────────────
    logging.info("계층2: RL 기준선 평가")

    if os.path.exists(args.t0_model):
        results.append(benchmark_t0_numerical(args.t0_model, env, device, N))
    else:
        logging.warning(f"T0 모델 없음: {args.t0_model} — 건너뜀")
        print(f"  ⚠ T0 모델 없음: {args.t0_model}")

    if args.mdp_p_path and os.path.exists(args.mdp_p_path):
        results.append(benchmark_kim_mdp_p(args.mdp_p_path, env, device, N))
    else:
        logging.info("김홍 MDP(P) 모델 없음 — 건너뜀")

    # ── 계층3: 본 연구 ────────────────────────────────────────────
    logging.info("계층3: 본 연구 모델 평가")

    if os.path.exists(args.lm_frozen):
        results.append(
            benchmark_lm_frozen(args.lm_frozen, env, device,
                                encoder, group_extractor, N,
                                args.from_checkpoint)
        )
    else:
        logging.warning(f"LM Frozen 모델 없음: {args.lm_frozen}")
        print(f"  ⚠ LM Frozen 모델 없음: {args.lm_frozen}")

    if os.path.exists(args.lm_finetuned):
        results.append(
            benchmark_lm_finetuned(args.lm_finetuned, env, device,
                                   encoder, group_extractor, N,
                                   args.from_checkpoint)
        )
    else:
        logging.warning(f"LM Fine-tuning 모델 없음: {args.lm_finetuned}")
        print(f"  ⚠ LM Fine-tuning 모델 없음: {args.lm_finetuned}")

    # ── 결과 출력 및 저장 ─────────────────────────────────────────
    print_benchmark_table(results)
    save_benchmark_csv(results)
    env.close()
    logging.info("벤치마크 완료.")


if __name__ == "__main__":
    main()