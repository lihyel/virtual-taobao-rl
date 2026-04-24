"""
=======================================================================
evaluate_lm_ddpg.py  —  LM-DDPG 최종 평가 스크립트
=======================================================================

[파일 위치]
  your_project/evaluate_lm_ddpg.py

[역할]
  학습 완료된 모델을 불러와 Greedy action만으로 대규모 평가 수행.
  교수님 피드백: "RL은 training 이후 test까지 해야 완료"

[평가 대상 (4종)]
  1. LM-DDPG (본 연구)         — Greedy action, DistilBERT 기반
  2. Random Policy              — 무작위 행동 (최하 기준선)
  3. Popularity Heuristic       — 항상 같은 인기 가중치 사용 (Rule-based)
  4. MDP-P 베이스라인           — 김홍(2023) 수치 기반 DDPG (있을 경우)

[김홍 evaluate.py 대비 변경점]
  - eval_mdp_p/g/mlp → eval_lm_ddpg: DistilBERT + textify 기반 추론
  - eval_random 유지
  - eval_popularity_heuristic 신규 추가 (Rule-based 비교군)
  - 결과를 CSV로 저장하여 논문 표 작성에 바로 활용 가능

[평가 지표]
  AvgReward  : 에피소드 누적 클릭 수 평균 (김홍 동일)
  CTR        : Σ클릭 / Σstep / 10 (김홍 동일)
  AvgSteps   : 평균 세션 길이 (이탈 전 step 수) — 본 연구 추가 지표
  ColdStartCTR: 첫 step(Cold-start) 시의 CTR — Cold-start 해결 핵심 지표

[실행 방법]
  python evaluate_lm_ddpg.py --model models/lm_actor_LM_DDPG_T1_final
  python evaluate_lm_ddpg.py --model models/checkpoint_ep500.pt --from_checkpoint
=======================================================================
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import gym
import virtualTB

from textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder, LMActor
from trainer_lm_ddpg import get_device


# =======================================================================
# [섹션 1] 환경 및 공통 설정
# =======================================================================

def make_env(seed: int = 42):
    env = gym.make('VirtualTB-v0')
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return env


# =======================================================================
# [섹션 2] LM-DDPG 로드 및 Greedy 평가
# =======================================================================

def load_lm_ddpg(actor_path: str, device: str, action_space,
                 encoder: DistilBERTEncoder) -> LMActor:
    """
    학습된 Actor 가중치 로드.

    평가 시에는 Critic이 필요 없음.
    Actor만 로드하여 greedy action을 수행.

    Args:
        actor_path: save_model()이 저장한 Actor state_dict 경로
                    또는 checkpoint_epN.pt (from_checkpoint=True 시)
    """
    actor = LMActor(
        hidden_size=128,
        action_space=action_space,
        bert_encoder=encoder,
    ).to(device)

    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()  # BatchNorm/Dropout 등 eval 모드
    logging.info(f"Actor 로드 완료: {actor_path}")
    return actor


def load_lm_ddpg_from_checkpoint(ckpt_path: str, device: str,
                                  action_space, encoder: DistilBERTEncoder) -> LMActor:
    """체크포인트(.pt)에서 Actor만 추출하여 로드."""
    ckpt = torch.load(ckpt_path, map_location=device)
    actor = LMActor(128, action_space, encoder).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    logging.info(f"체크포인트에서 Actor 로드: {ckpt_path} (Ep {ckpt['episode']})")
    return actor


def greedy_action(actor: LMActor, prompt: str,
                  dynamic: np.ndarray, device: str) -> np.ndarray:
    """
    Greedy action 선택 (noise 없음).

    학습 중 exploration(OUNoise)과 달리 테스트에서는
    Actor의 출력을 그대로 사용.
    """
    dyn_t = torch.FloatTensor(dynamic).unsqueeze(0).to(device)
    with torch.no_grad():
        action = actor([prompt], dyn_t).squeeze(0)
    return action.clamp(-1, 1).cpu().numpy()


def eval_lm_ddpg(actor: LMActor, env, group_extractor,
                 device: str, n_episodes: int = 5000) -> dict:
    """
    본 연구 LM-DDPG 평가

    김홍 evaluate.py의 eval_mdp_p()와 동일한 구조,
    state 처리만 textify 기반으로 교체.

    Args:
        n_episodes: 김홍(2023) 평가 기준 5000 에피소드
    """
    total_reward = 0
    total_steps  = 0
    cold_reward  = 0   # Cold-start(step=0) 이후 첫 reward
    cold_count   = 0

    for _ in tqdm(range(n_episodes), desc="[LM-DDPG 평가]"):
        state_91d = env.reset()
        ep_reward = 0
        step = 0

        while True:
            res    = prepare_state(state_91d, group_extractor)
            prompt = res["prompt"]
            dyn    = res["dynamic_numeric"]
            is_cold = res["is_cold_start"]

            # Greedy action (noise 없음) — 핵심 차이
            action = greedy_action(actor, prompt, dyn, device)

            next_state, reward, done, _ = env.step(action)

            # Cold-start 첫 번째 추천 reward 별도 기록
            if is_cold:
                cold_reward += reward
                cold_count  += 1

            ep_reward    += reward
            total_steps  += 1
            step         += 1
            state_91d     = next_state

            if done:
                break

        total_reward += ep_reward

    avg_reward = total_reward / n_episodes
    ctr        = total_reward / total_steps / 10
    avg_steps  = total_steps / n_episodes
    cold_ctr   = cold_reward / cold_count / 10 if cold_count > 0 else 0

    return {
        "model": "LM-DDPG (본 연구)",
        "AvgReward":     round(avg_reward, 4),
        "CTR":           round(ctr, 4),
        "AvgSteps":      round(avg_steps, 2),
        "ColdStartCTR":  round(cold_ctr, 4),
        "n_episodes":    n_episodes,
    }


# =======================================================================
# [섹션 3] Random Policy 평가
# =======================================================================

def eval_random(env, n_episodes: int = 5000) -> dict:
    """
    무작위 행동 정책.

    김홍 evaluate.py의 eval_random()과 동일.
    VT action space: 27D 연속 벡터 ∈ [-1, 1] → 균등 분포 샘플링

    목적: 최하 기준선 (아무것도 안 하는 것 대비 LM의 기여 확인)
    """
    total_reward = 0
    total_steps  = 0
    cold_reward  = 0
    cold_count   = 0

    for _ in tqdm(range(n_episodes), desc="[Random 평가]"):
        state = env.reset()
        t = 0

        while True:
            # 27D 균등 랜덤 행동 (김홍 eval_random과 동일)
            action = (np.random.rand(27) - 0.5) * 2

            next_state, reward, done, _ = env.step(action)

            if t == 0:  # Cold-start 첫 추천
                cold_reward += reward
                cold_count  += 1

            total_reward += reward
            total_steps  += 1
            t += 1
            state = next_state
            if done:
                break

    return {
        "model":        "Random Policy",
        "AvgReward":    round(total_reward / n_episodes, 4),
        "CTR":          round(total_reward / total_steps / 10, 4),
        "AvgSteps":     round(total_steps / n_episodes, 2),
        "ColdStartCTR": round(cold_reward / cold_count / 10 if cold_count > 0 else 0, 4),
        "n_episodes":   n_episodes,
    }


# =======================================================================
# [섹션 4] Popularity Heuristic (Rule-based)
# =======================================================================

def eval_popularity_heuristic(env, n_episodes: int = 5000) -> dict:
    """
    인기도 기반 휴리스틱 정책 (Rule-based, 강화학습 아님).

    설계 근거:
      실제 추천 시스템에서 가장 기본적인 non-ML 베이스라인은
      "항상 가장 인기 있는 아이템을 추천"하는 것입니다.
      VirtualTaobao에서 "인기"를 직접 관찰할 수 없으므로,
      과거 에피소드에서 reward가 높았던 방향의 action weight를
      고정값으로 사용하는 방식으로 근사합니다.

      구체적으로: 모든 가중치를 양수 최대값(1.0)으로 고정
      → "어떤 아이템이든 선호" = 가장 보수적인 인기도 추천

    이 정책이 RL보다 낮으면 → RL이 실제로 선택적 추천을 학습했음을 증명
    이 정책이 RL보다 높으면 → RL 학습이 불충분하다는 신호 (문제 진단용)
    """
    # 모든 가중치를 1로 고정 (항상 최대 선호 시그널)
    fixed_action = np.ones(27, dtype=np.float32)

    total_reward = 0
    total_steps  = 0
    cold_reward  = 0
    cold_count   = 0

    for _ in tqdm(range(n_episodes), desc="[Popularity Heuristic 평가]"):
        state = env.reset()
        t = 0

        while True:
            next_state, reward, done, _ = env.step(fixed_action)

            if t == 0:
                cold_reward += reward
                cold_count  += 1

            total_reward += reward
            total_steps  += 1
            t += 1
            state = next_state
            if done:
                break

    return {
        "model":        "Popularity Heuristic (Rule-based)",
        "AvgReward":    round(total_reward / n_episodes, 4),
        "CTR":          round(total_reward / total_steps / 10, 4),
        "AvgSteps":     round(total_steps / n_episodes, 2),
        "ColdStartCTR": round(cold_reward / cold_count / 10 if cold_count > 0 else 0, 4),
        "n_episodes":   n_episodes,
    }


# =======================================================================
# [섹션 5] 수치 기반 DDPG 평가 (있을 경우 — 김홍 재현)
# =======================================================================

class _NumericalActor(nn.Module):
    """
    T0 수치 기반 DDPG의 Actor 네트워크.

    exp3_t0_baseline.py와 동일한 구조를 인라인으로 정의.
    외부 파일 의존성을 없애 Pylance 경고 및 임포트 에러를 방지.

    구조: Linear(91→128) + LN + ReLU → Linear(128→128) + LN + ReLU → Linear(128→27) + Tanh
    """
    class _LayerNorm(nn.Module):
        def __init__(self, n, eps=1e-5):
            super().__init__()
            self.gamma = nn.Parameter(torch.ones(n))
            self.beta  = nn.Parameter(torch.zeros(n))
            self.eps   = eps
        def forward(self, x):
            mean = x.mean(-1, keepdim=True)
            std  = x.std(-1, keepdim=True)
            return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def __init__(self, num_inputs: int, hidden_size: int, num_outputs: int):
        super().__init__()
        LN = _NumericalActor._LayerNorm
        self.l1  = nn.Linear(num_inputs, hidden_size)
        self.ln1 = LN(hidden_size)
        self.l2  = nn.Linear(hidden_size, hidden_size)
        self.ln2 = LN(hidden_size)
        self.mu  = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        return torch.tanh(self.mu(x))


def eval_numerical_ddpg(actor_path: str, env, device: str,
                        n_episodes: int = 5000) -> dict:
    """
    T0 수치 기반 DDPG 평가 (exp3_t0_baseline.py 학습 결과).

    김홍 evaluate.py의 eval_mdp_p()와 동일한 방식.
    LM 없이 91D 수치 직접 입력.
    LM-DDPG와의 비교를 통해 LM 도입 효과를 정량적으로 제시.

    [수정] from exp3_t0_baseline import Actor 제거
           → _NumericalActor 클래스를 이 파일 안에 직접 정의
           → 외부 파일 의존성 없음, Pylance 경고 해결
    """
    num_inputs  = env.observation_space.shape[0]  # 91
    num_outputs = env.action_space.shape[0]        # 27

    # Actor 구조: exp3_t0_baseline.py의 Actor와 동일 (인라인 정의 사용)
    actor = _NumericalActor(num_inputs, 128, num_outputs).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()

    total_reward = 0
    total_steps  = 0
    cold_reward  = 0
    cold_count   = 0

    for _ in tqdm(range(n_episodes), desc="[T0 수치 DDPG 평가]"):
        state = torch.FloatTensor([env.reset()]).to(device)
        t = 0

        while True:
            with torch.no_grad():
                action = actor(state).clamp(-1, 1).cpu().numpy().squeeze()

            next_state_raw, reward, done, _ = env.step(action)

            if t == 0:
                cold_reward += reward
                cold_count  += 1

            total_reward += reward
            total_steps  += 1
            t += 1
            state = torch.FloatTensor([next_state_raw]).to(device)
            if done:
                break

    return {
        "model":        "T0 Numerical DDPG (김홍 재현)",
        "AvgReward":    round(total_reward / n_episodes, 4),
        "CTR":          round(total_reward / total_steps / 10, 4),
        "AvgSteps":     round(total_steps / n_episodes, 2),
        "ColdStartCTR": round(cold_reward / cold_count / 10 if cold_count > 0 else 0, 4),
        "n_episodes":   n_episodes,
    }


# =======================================================================
# [섹션 6] 결과 출력 및 저장
# =======================================================================

def print_results(results: list):
    """평가 결과를 논문 표 형식으로 출력."""
    print()
    print("=" * 75)
    print("【최종 평가 결과】")
    print("=" * 75)
    header = f"{'모델':<35} {'AvgReward':>10} {'CTR':>8} {'AvgSteps':>10} {'ColdCTR':>10}"
    print(header)
    print("-" * 75)

    # 김홍 SOTA 기준선 (비교용 표시)
    print(f"  {'[기준] 김홍 MDP(P)+G+MLP SOTA':<33} {'56.54':>10} {'0.767':>8} {'-':>10} {'-':>10}")
    print("-" * 75)

    for r in results:
        row = (f"  {r['model']:<33} "
               f"{r['AvgReward']:>10.4f} "
               f"{r['CTR']:>8.4f} "
               f"{r['AvgSteps']:>10.2f} "
               f"{r['ColdStartCTR']:>10.4f}")
        print(row)

    print("=" * 75)
    print()
    print("지표 설명:")
    print("  AvgReward  : 에피소드 평균 누적 클릭 수 (김홍 동일)")
    print("  CTR        : Σ클릭 / Σstep / 10 (김홍 동일)")
    print("  AvgSteps   : 평균 세션 길이 (이탈 전 추천 횟수)")
    print("  ColdCTR    : 첫 추천(Cold-start)에서의 CTR")


def save_results_csv(results: list, path: str = "output/eval_results.csv"):
    """결과를 CSV로 저장 — 논문 표 작성에 직접 활용."""
    import csv
    os.makedirs("output", exist_ok=True)
    keys = ["model", "AvgReward", "CTR", "AvgSteps", "ColdStartCTR", "n_episodes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    logging.info(f"결과 저장: {path}")


# =======================================================================
# [섹션 7] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description="LM-DDPG 최종 평가")
    parser.add_argument("--model",
                        default="models/lm_actor_LM_DDPG_T1_final",
                        help="학습된 Actor 가중치 경로")
    parser.add_argument("--from_checkpoint", action="store_true",
                        help="모델 경로가 checkpoint_epN.pt 형식일 때 사용")
    parser.add_argument("--t0_model", default=None,
                        help="T0 수치 DDPG 모델 경로 (있으면 추가 평가)")
    parser.add_argument("--n_episodes", type=int, default=5000,
                        help="평가 에피소드 수 (기본 3000 / 김홍 기준 5000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("eval_results.log"),
        ]
    )

    device = get_device()
    env    = make_env(seed=args.seed)
    N      = args.n_episodes

    logging.info(f"평가 시작 | 에피소드: {N} | 디바이스: {device}")

    # ── 집단 정보 로드 (학습 때와 동일 K-Means) ──────────────────────
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    # ── DistilBERT 인코더 (추론 시 frozen — 빠름) ──────────────────────
    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)

    # ── LM-DDPG Actor 로드 ────────────────────────────────────────────
    if args.from_checkpoint:
        actor = load_lm_ddpg_from_checkpoint(args.model, device,
                                              env.action_space, encoder)
    else:
        actor = load_lm_ddpg(args.model, device, env.action_space, encoder)

    # ── 평가 실행 ─────────────────────────────────────────────────────
    results = []

    # 1) 본 연구 LM-DDPG
    logging.info("=== LM-DDPG (본 연구) 평가 시작 ===")
    results.append(eval_lm_ddpg(actor, env, group_extractor, device, N))

    # 2) Random Policy (최하 기준선)
    logging.info("=== Random Policy 평가 시작 ===")
    results.append(eval_random(env, N))

    # 3) Popularity Heuristic (Rule-based 비교군)
    logging.info("=== Popularity Heuristic 평가 시작 ===")
    results.append(eval_popularity_heuristic(env, N))

    # 4) T0 수치 DDPG (있을 경우)
    if args.t0_model and os.path.exists(args.t0_model):
        logging.info("=== T0 수치 DDPG 평가 시작 ===")
        results.append(eval_numerical_ddpg(args.t0_model, env, device, N))
    else:
        logging.info("T0 모델 없음 — 수치 DDPG 평가 건너뜀")

    # ── 결과 출력 및 저장 ─────────────────────────────────────────────
    print_results(results)
    save_results_csv(results)
    env.close()
    logging.info("평가 완료.")


if __name__ == "__main__":
    main()