import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화 스타일 설정 (학회 발표용)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_presentation_graphs():
    # 1. 데이터 로드 (trainer_lm_ddpg.py에서 저장한 파일들)
    rewards = np.load("output/lm_ddpg_rewards.npy")
    val_ctr = np.load("output/lm_ddpg_val_ctr.npy")
    val_episodes = np.arange(0, len(val_ctr) * 20, 20) # 20 에피소드마다 검증했다는 가정

    # --- 그래프 1: Cumulative Reward (학습 수렴 곡선) ---
    plt.figure(figsize=(10, 5))
    # 이동 평균(Moving Average)으로 노이즈 제거 (선배님 논문 방식)
    window = 50
    smooth_reward = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    plt.plot(rewards, alpha=0.2, color='royalblue', label='Raw Reward')
    plt.plot(smooth_reward, color='navy', linewidth=2, label=f'{window}-Ep Moving Avg')
    plt.title("Training Convergence: LM-DDPG", fontsize=15, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Cumulative Reward", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/presentation_reward_curve.png", dpi=300)
    plt.show()

    # --- 그래프 2: CTR Performance (막대 그래프) ---
    # 실제 테스트 결과 수치로 교체하세요 (예시 데이터)
    model_names = ['Random', 'MDP+G+MLP', 'LM-DDPG (Ours)']
    # 모델의 실제 CTR 수치를
    ctr_values = [0.0022, 0.7061, 0.9998]
    
    plt.figure(figsize=(8, 6))
    colors = ['lightgray', 'darkgray', 'royalblue']
    bars = plt.bar(model_names, ctr_values, color=colors, edgecolor='black', linewidth=1.2)
    
    # 막대 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        # 0.9998 -> 99.98% 로 표시
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f'{height*100:.2f}%', ha='center', fontweight='bold')

    plt.title("Comparative Analysis: CTR Performance", fontsize=15, fontweight='bold')
    plt.ylabel("Click-Through Rate (%)")
    plt.ylim(0, 1.1) # Y축 범위를 110% 정도로 여유 있게 설정
    plt.tight_layout()
    plt.savefig("output/presentation_ctr_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_presentation_graphs()