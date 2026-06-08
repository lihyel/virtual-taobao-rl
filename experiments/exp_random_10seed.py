"""
=======================================================================
experiments/exp_random_10seed.py  —  Random Policy 정량적 검증 (10 Seed)
=======================================================================

[실험 목적]
  Random Policy 기준선의 통계적으로 신뢰할 수 있는 성능 수치 도출.
  10개 seed × 5000 에피소드 → mean ± std 형태로 보고.

[Random Policy 정의]
  action = (np.random.rand(27) - 0.5) * 2  ← 김홍 eval_random()과 동일
  학습 없음. seed별로 np.random.seed()만 다르게 설정.

[evaluate_benchmark_v2.py와의 관계]
  - run_episodes(), benchmark_random() 재사용
  - T0/LM 관련 함수는 사용하지 않음 (이 파일에서 제외)
  - evaluate_benchmark_v2.py는 단일 seed 단발성 평가용
  - 이 파일은 10 seed 반복 실험 + mean ± std 도출 전용

[출력 파일]
  output/exp_random_10seed/
  ├── seed_{N}/
  │   └── eval_result.csv
  ├── exp_random_10seed.log
  ├── all_seeds_summary.csv    ← mean ± std 포함 (논문 Table용)
  └── random_seed_bar.png      ← seed별 결과 시각화

[실행]
  # GPU 지정 실행 (Random Policy는 GPU 불필요하지만 통일성 위해)
  python experiments/exp_random_10seed.py --gpu 0
  python experiments/exp_random_10seed.py --seeds 0 1 2 3 4 5 6 7 8 9
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gym
import virtualTB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =======================================================================
# [섹션 1] GPU 선택 (exp_T1_10seed.py와 동일 방식)
# =======================================================================

def get_device_with_id(gpu_id=None) -> str:
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.seed(seed)


# =======================================================================
# [섹션 2] 평가 루프 (evaluate_benchmark_v2.py의 run_episodes 재사용)
# =======================================================================

def run_episodes(env, action_fn, n_episodes: int, desc: str) -> dict:
    """
    evaluate_benchmark_v2.py의 run_episodes와 동일한 구조.
    AvgReward(본연구 방식)와 CTR만 사용 (AvgReward_KimHong 제외 — 맥락 다름).
    """
    total_reward = 0.0
    total_steps  = 0
    cold_reward  = 0.0
    cold_count   = 0

    for _ in tqdm(range(n_episodes), desc=f"  {desc:<30}", leave=False):
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

    avg_reward = total_reward / n_episodes
    ctr        = total_reward / total_steps / 10 if total_steps > 0 else 0
    avg_steps  = total_steps  / n_episodes
    cold_ctr   = cold_reward  / cold_count  / 10 if cold_count > 0 else 0

    return {
        "AvgReward":    round(avg_reward, 4),
        "CTR":          round(ctr, 4),
        "AvgSteps":     round(avg_steps, 2),
        "ColdStartCTR": round(cold_ctr, 4),
        "n_episodes":   n_episodes,
    }


# =======================================================================
# [섹션 3] seed별 평가
# =======================================================================

def evaluate_one_seed(seed: int, n_eval: int, output_dir: str) -> dict:
    """
    단일 seed로 Random Policy 평가.
    학습 없음. seed만 고정하고 random action 생성.
    """
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make('VirtualTB-v0')
    set_global_seed(seed, env)

    def action_fn(state_91d):
        return (np.random.rand(27) - 0.5) * 2   # 김홍 eval_random()과 동일

    metrics = run_episodes(env, action_fn, n_eval, f"Random | Seed {seed}")
    env.close()

    result = {
        "model":        "Random Policy",
        "seed":         seed,
        "AvgReward":    metrics["AvgReward"],
        "CTR":          metrics["CTR"],
        "ColdStartCTR": metrics["ColdStartCTR"],
        "AvgSteps":     metrics["AvgSteps"],
        "n_eval":       n_eval,
    }

    with open(f"{output_dir}/eval_result.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

    logging.info(
        f"  [Random | Seed {seed}] "
        f"AvgReward: {result['AvgReward']:.4f} | "
        f"CTR: {result['CTR']:.4f} | "
        f"ColdCTR: {result['ColdStartCTR']:.4f}"
    )
    return result


# =======================================================================
# [섹션 4] 결과 집계 및 시각화
# =======================================================================

def save_summary_csv(results: list, output_dir: str):
    path = f"{output_dir}/all_seeds_summary.csv"
    fields = ["model", "seed", "AvgReward", "CTR", "ColdStartCTR",
              "AvgSteps", "n_eval"]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

        # mean ± std 행 추가
        for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
            vals = [r[m] for r in results]
            writer.writerow({
                "model": "mean±std", "seed": "─",
                m: f"{np.mean(vals):.4f}±{np.std(vals):.4f}",
            })

    logging.info(f"요약 CSV 저장: {path}")

    # 콘솔 출력
    print(f"\n{'='*60}")
    print("  Random Policy 정량적 검증 결과 (10 Seed)")
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


def plot_seed_results(results: list, output_dir: str):
    """seed별 결과 막대 그래프 + mean ± std 시각화."""
    metrics = ["AvgReward", "CTR", "ColdStartCTR"]
    seeds   = [r["seed"] for r in results]
    colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(seeds)))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Random Policy: 10-Seed Quantitative Validation",
                 fontsize=12)

    for ax, metric in zip(axes, metrics):
        vals  = [r[metric] for r in results]
        mean_ = np.mean(vals)
        std_  = np.std(vals)

        bars = ax.bar([f"s{s}" for s in seeds], vals,
                      color=colors, alpha=0.85)
        ax.axhline(mean_, color='red', linewidth=1.8,
                   linestyle='--', label=f"Mean: {mean_:.4f}")
        ax.axhspan(mean_ - std_, mean_ + std_,
                   alpha=0.12, color='red', label=f"±Std: {std_:.4f}")

        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("Seed", fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/random_seed_bar.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    logging.info(f"시각화 저장: {output_dir}/random_seed_bar.png")


# =======================================================================
# [섹션 5] 메인
# =======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Random Policy 정량적 검증 — 10 Seed 반복실험"
    )
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(range(10)),
                        help="실험 seed 목록 (기본: 0~9)")
    parser.add_argument("--n_eval",     type=int, default=5000,
                        help="평가 에피소드 수 (기본: 5000, 김홍과 동일)")
    parser.add_argument("--gpu",        type=int, default=None,
                        help="사용할 GPU 번호 (Random은 GPU 불필요, 통일성 위해)")
    parser.add_argument("--output_dir", default="output/exp_random_10seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/exp_random_10seed.log"),
        ],
    )

    # Random Policy는 GPU 사용 안 하지만 인자 구조 통일
    device = get_device_with_id(args.gpu)

    logging.info("="*55)
    logging.info("Random Policy 정량적 검증 실험 시작")
    logging.info(f"  Seeds:     {args.seeds}  (총 {len(args.seeds)}개)")
    logging.info(f"  평가 ep:   {args.n_eval}")
    logging.info(f"  저장 위치: {args.output_dir}")
    logging.info(f"  디바이스:  {device}  (Random은 CPU 연산)")
    logging.info("="*55)

    all_results = []

    for seed in args.seeds:
        seed_dir = f"{args.output_dir}/seed_{seed}"
        result   = evaluate_one_seed(seed, args.n_eval, seed_dir)
        all_results.append(result)

    # ── 결과 집계 + 시각화 ────────────────────────────────────────────
    save_summary_csv(all_results, args.output_dir)
    plot_seed_results(all_results, args.output_dir)

    # ── 최종 mean ± std 로그 ──────────────────────────────────────────
    logging.info("="*55)
    logging.info("Random Policy 최종 결과 (mean ± std)")
    logging.info("="*55)
    for m in ["AvgReward", "CTR", "ColdStartCTR", "AvgSteps"]:
        vals = [r[m] for r in all_results]
        logging.info(
            f"  {m:<15}: {np.mean(vals):.4f} ± {np.std(vals):.4f}"
            f"  (min={min(vals):.4f}, max={max(vals):.4f})"
        )
    logging.info("="*55)
    logging.info(f"실험 완료. 결과: {args.output_dir}/")


if __name__ == "__main__":
    main()