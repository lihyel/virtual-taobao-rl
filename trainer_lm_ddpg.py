"""
=======================================================================
trainer_lm_ddpg.py  ─  LM 기반 DDPG 학습 루프 (v2)
=======================================================================

[v1 → v2 수정 내역]
  1. UserWarning 수정
     Before: torch.FloatTensor(dynamics)  ← list of numpy → 느림
     After:  torch.FloatTensor(np.array(dynamics))  ← numpy 먼저 stack

  2. M1 Mac GPU 지원 (MPS: Metal Performance Shaders)
     device 우선순위: mps > cuda > cpu
     → 맥북 Pro 14 M1에서 GPU 가속 활성화

  3. 500 에피소드마다 체크포인트 자동 저장
     → 예기치 못한 종료 시 마지막 체크포인트부터 재개

  4. resume 기능
     → main(resume=True)로 호출 시 가장 최근 체크포인트 자동 로드

  5. OUNoise.reset() 에피소드마다 호출
     → 탐색 노이즈 초기화 누락 버그 수정

  6. 김홍(2023) 검증 루프 추가
     → 20 에피소드마다 50개 에피소드 평가, CTR 출력

  7. 저장 경로 구조 정리
     models/checkpoint_ep{N}.pt  ← 에피소드별 체크포인트
     models/lm_ddpg_final.*      ← 최종 모델
     output/lm_ddpg_*.npy        ← 학습 곡선

[파일 위치]
  your_project/trainer_lm_ddpg.py

[실행 방법]
  처음 학습:    python trainer_lm_ddpg.py
  이어서 학습:  python trainer_lm_ddpg.py --resume

[Docker 환경]
  device는 자동 감지. 서버(NVIDIA GPU)에서는 "cuda"로 자동 전환.
=======================================================================
"""

import os
import sys
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

from textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder, LMActor, LMCritic


# =======================================================================
# [섹션 1] 디바이스 설정
# =======================================================================

def get_device() -> str:
    """
    실행 환경에 맞는 최적 디바이스 자동 감지.

    우선순위: MPS (M1/M2 Mac) > CUDA (서버/Docker) > CPU
    - MPS: 맥북 Pro 14 M1에서 Metal GPU 가속
    - CUDA: Docker 서버 환경
    - CPU: 그 외 (fallback)
    """
    if torch.backends.mps.is_available():
        logging.info("디바이스: MPS (M1 Mac GPU)")
        return "mps"
    elif torch.cuda.is_available():
        logging.info("디바이스: CUDA")
        return "cuda"
    else:
        logging.info("디바이스: CPU")
        return "cpu"


# =======================================================================
<<<<<<< HEAD
# [섹션 2] Replay Memory — 91D 원본 저장 방식
=======
# [섹션 2] Replay Memory — 91D 원본 저장 방식 (혜리님 설계 유지)
>>>>>>> c81a657868e8876a5941fe31d9bb3d86081f00f3
# =======================================================================
# 설계 근거: 91D 수치를 메모리에 저장 후 배치 학습 시 일괄 텍스트화
# → 메모리 효율적 (프롬프트 문자열 대비 용량 절약)
# → update_parameters()에서 배치 단위로 prepare_state() 호출

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


# =======================================================================
# [섹션 3] OUNoise — 김홍(2023)과 동일
# =======================================================================

class OUNoise:
    """Ornstein-Uhlenbeck 탐색 노이즈."""
    def __init__(self, action_dim, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        self.reset()

    def reset(self):
        # [버그 수정] 에피소드마다 reset() 필수 — v1에서 누락됨
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale


# =======================================================================
# [섹션 4] LMDDPG 에이전트
# =======================================================================

class LMDDPG:
    """
    LM 기반 DDPG 에이전트.
    혜리님 v1 코드 구조를 유지하며 버그/성능 수정만 반영.
    """

    def __init__(self, gamma, tau, hidden_size, action_space, encoder, device):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor         = LMActor(hidden_size, action_space, encoder).to(device)
        self.actor_target  = LMActor(hidden_size, action_space, encoder).to(device)
        # [중요] DistilBERT fine-tuning lr 차등 적용
        # encoder 파라미터: 1e-5 (사전학습 보존)
        # FC 레이어:         1e-4 (빠른 수렴)
        encoder_param_ids = {id(p) for p in encoder.parameters()}
        actor_fc_params   = [p for p in self.actor.parameters()
                             if id(p) not in encoder_param_ids]
        self.actor_optim  = torch.optim.Adam([
            {"params": encoder.parameters(), "lr": 1e-5},
            {"params": actor_fc_params,       "lr": 1e-4},
        ])

        self.critic         = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_target  = LMCritic(hidden_size, action_space, encoder).to(device)
        self.critic_optim   = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # 타겟 네트워크 하드 초기화
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, prompt: str, dynamic_numeric: torch.Tensor) -> torch.Tensor:
        """
        프롬프트 + 동적수치 → 27D action.
        Args:
            prompt         : textify.py 출력 문자열
            dynamic_numeric: (1, 3) torch.Tensor (배치 차원 포함)
        """
        self.actor.eval()
        with torch.no_grad():
            mu = self.actor(prompt, dynamic_numeric)
        self.actor.train()
        return mu.clamp(-1, 1)

    def update_parameters(
        self,
        batch,
        group_extractor: GroupInfoExtractor,
    ) -> tuple:
        """
        배치 학습. 91D 원본 → 배치 단위 텍스트화 후 업데이트.

        [v1 버그 수정]
        Before: torch.FloatTensor(dynamics)  ← list of numpy → UserWarning
        After:  torch.FloatTensor(np.array(dynamics))  ← numpy stack 먼저
        """
        # ── 배치 텍스트화 ────────────────────────────────────────────
        prompts, dynamics, next_prompts, next_dynamics = [], [], [], []

        for s, ns in zip(batch.state_91d, batch.next_state_91d):
            r  = prepare_state(s,  group_extractor)
            nr = prepare_state(ns, group_extractor)
            prompts.append(r['prompt'])
            dynamics.append(r['dynamic_numeric'])
            next_prompts.append(nr['prompt'])
            next_dynamics.append(nr['dynamic_numeric'])

        # [수정] list of numpy → np.array()로 먼저 stack → UserWarning 제거
        dyn_t      = torch.FloatTensor(np.array(dynamics)).to(self.device)
        next_dyn_t = torch.FloatTensor(np.array(next_dynamics)).to(self.device)

        action_t = torch.cat(list(batch.action)).to(self.device)
        reward_t = torch.cat(list(batch.reward)).unsqueeze(1).to(self.device)
        mask_t   = torch.cat(list(batch.mask)).unsqueeze(1).to(self.device)

        # ── Critic 업데이트 (Bellman equation) ───────────────────────
        with torch.no_grad():
            next_act = self.actor_target(next_prompts, next_dyn_t)
            next_q   = self.critic_target(next_prompts, next_dyn_t, next_act)
            target_q = reward_t + (self.gamma * mask_t * next_q)

        curr_q     = self.critic(prompts, dyn_t, action_t)
        value_loss = F.mse_loss(curr_q, target_q)

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        # ── Actor 업데이트 (Policy gradient) ─────────────────────────
        policy_loss = -self.critic(
            prompts, dyn_t, self.actor(prompts, dyn_t)
        ).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # ── Soft target update ────────────────────────────────────────
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)

        return value_loss.item(), policy_loss.item()

    # ── 체크포인트 저장/로드 ────────────────────────────────────────

    def save_checkpoint(self, episode: int, rewards: list, val_rewards: list,
                        val_ctr: list, value_losses: list, policy_losses: list):
        """
        체크포인트 저장.
        학습 상태(에피소드, 보상 기록)와 모델 가중치를 함께 저장.
        → 예기치 못한 종료 시 이 파일로 재개 가능.
        """
        os.makedirs("models", exist_ok=True)
        path = f"models/checkpoint_ep{episode}.pt"
        torch.save({
            "episode":       episode,
            "actor_state":   self.actor.state_dict(),
            "critic_state":  self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optim":   self.actor_optim.state_dict(),
            "critic_optim":  self.critic_optim.state_dict(),
            "rewards":        rewards,
            "val_rewards":    val_rewards,
            "val_ctr":        val_ctr,
            "value_losses":   value_losses,
            "policy_losses":  policy_losses,
        }, path)
        logging.info(f"[체크포인트] 저장 완료: {path}")
        return path

    def load_checkpoint(self, path: str) -> dict:
        """
        체크포인트 로드. 저장된 학습 상태 딕셔너리 반환.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor_state"])
        self.critic.load_state_dict(ckpt["critic_state"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        logging.info(f"[체크포인트] 로드 완료: {path} (에피소드 {ckpt['episode']}부터 재개)")
        return ckpt

    def save_model(self, suffix: str = "final"):
        """최종 모델 저장."""
        os.makedirs("models", exist_ok=True)
        torch.save(self.actor.state_dict(),  f"models/lm_actor_{suffix}")
        torch.save(self.critic.state_dict(), f"models/lm_critic_{suffix}")
        logging.info(f"[최종 모델] 저장: models/lm_actor_{suffix}, models/lm_critic_{suffix}")


# =======================================================================
# [섹션 5] 최신 체크포인트 탐색 유틸
# =======================================================================

def find_latest_checkpoint() -> Optional[str]:
    """
    models/ 디렉토리에서 가장 최근 체크포인트 파일을 반환.
    파일명 형식: checkpoint_ep{N}.pt
    """
    if not os.path.isdir("models"):
        return None
    ckpts = [f for f in os.listdir("models")
             if f.startswith("checkpoint_ep") and f.endswith(".pt")]
    if not ckpts:
        return None
    # 에피소드 번호 기준 정렬
    ckpts.sort(key=lambda x: int(x.replace("checkpoint_ep", "").replace(".pt", "")))
    return os.path.join("models", ckpts[-1])


# =======================================================================
# [섹션 6] 메인 학습 루프
# =======================================================================

def main(resume: bool = False):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),  # 로그 파일도 저장
        ]
    )

    # ── 디바이스 ──────────────────────────────────────────────────────
    device = get_device()

    # ── 환경 초기화 ───────────────────────────────────────────────────
    env = gym.make('VirtualTB-v0')
    env.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # ── K-Means 집단 정보 ─────────────────────────────────────────────
    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",
    )
    group_extractor.fit()

    # ── DistilBERT 인코더 ─────────────────────────────────────────────
    # M1 Mac 로컬: trainable=False (frozen) → 속도 우선
    # Docker 서버: trainable=True (fine-tuning) → 성능 우선
    is_local_mac = (device == "mps")
    encoder = DistilBERTEncoder(
        trainable=not is_local_mac,   # Mac=frozen, 서버=fine-tuning
        device=device,
        max_length=128,
    )
    if is_local_mac:
        logging.info("M1 Mac 환경: DistilBERT frozen (추론 속도 최적화)")
    else:
        logging.info("서버 환경: DistilBERT fine-tuning 활성화")

    # ── LMDDPG 에이전트 ────────────────────────────────────────────────
    agent = LMDDPG(
        gamma=0.7,
        tau=0.003,       # 혜리님 v1 값 유지
        hidden_size=128,
        action_space=env.action_space,
        encoder=encoder,
        device=device,
    )

    # ── 학습 상태 초기화 ───────────────────────────────────────────────
    memory      = ReplayMemory(10000)
    ounoise     = OUNoise(env.action_space.shape[0])
    batch_size  = 32
    n_episodes  = 2000
    start_ep    = 0
    checkpoint_interval = 500  # 500 에피소드마다 자동 저장

    rewards, val_rewards, val_ctr = [], [], []
    value_losses, policy_losses   = [], []

    # ── 재개 (resume) ──────────────────────────────────────────────────
    if resume:
        ckpt_path = find_latest_checkpoint()
        if ckpt_path:
            ckpt = agent.load_checkpoint(ckpt_path)
            start_ep      = ckpt["episode"] + 1
            rewards       = ckpt["rewards"]
            val_rewards   = ckpt["val_rewards"]
            val_ctr       = ckpt["val_ctr"]
            value_losses  = ckpt["value_losses"]
            policy_losses = ckpt["policy_losses"]
            logging.info(f"에피소드 {start_ep}부터 재개합니다.")
        else:
            logging.warning("체크포인트를 찾지 못했습니다. 처음부터 시작합니다.")

    # ── 학습 루프 ─────────────────────────────────────────────────────
    for i_ep in range(start_ep, n_episodes):

        state_91d = env.reset()
        ounoise.reset()   # [버그 수정] 에피소드 시작 시 노이즈 초기화
        episode_reward = 0

        while True:
            # 1. 텍스트화
            res    = prepare_state(state_91d, group_extractor)
            prompt = res['prompt']
            dyn    = torch.FloatTensor(res['dynamic_numeric']).unsqueeze(0).to(device)

            # 2. 액션 선택 (OUNoise 탐색)
            action_t = agent.select_action(prompt, dyn)
            # OUNoise는 CPU numpy에서 계산 후 더하기
            noise    = torch.FloatTensor(ounoise.noise()).to(device)
            action_t = (action_t + noise).clamp(-1, 1)

            next_state_91d, reward, done, _ = env.step(action_t.cpu().numpy().squeeze())

            # 3. 메모리 저장 (91D 원본 저장 — 배치 학습 시 텍스트화)
            memory.push(
                state_91d,
                action_t.detach().cpu().unsqueeze(0),   # (1, 27)
                torch.Tensor([not done]),
                next_state_91d,
                torch.Tensor([reward]),
            )

            state_91d = next_state_91d
            episode_reward += reward

            # 4. 배치 학습
            if len(memory) > batch_size:
                for _ in range(5):   # 김홍 동일: 매 step 5회 업데이트
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    v_loss, p_loss = agent.update_parameters(batch, group_extractor)
                    value_losses.append(v_loss)
                    policy_losses.append(p_loss)

            if done:
                break

        rewards.append(episode_reward)
        logging.info(f"Episode {i_ep:4d} | Reward: {episode_reward:.1f}")

        # ── 검증 (20 에피소드마다) ─────────────────────────────────
        if i_ep > 0 and i_ep % 20 == 0:
            val_r, val_s = 0, 0
            for _ in range(50):
                s = env.reset()
                r_ep = prepare_state(s, group_extractor)
                p, d = r_ep['prompt'], torch.FloatTensor(r_ep['dynamic_numeric']).unsqueeze(0).to(device)
                while True:
                    a = agent.select_action(p, d).cpu().numpy().squeeze()
                    ns, r, done, _ = env.step(a)
                    val_r += r
                    val_s += 1
                    nr = prepare_state(ns, group_extractor)
                    p  = nr['prompt']
                    d  = torch.FloatTensor(nr['dynamic_numeric']).unsqueeze(0).to(device)
                    if done:
                        break

            avg_r = val_r / 50
            ctr   = val_r / val_s / 10 if val_s > 0 else 0
            val_rewards.append(avg_r)
            val_ctr.append(ctr)
            vl = value_losses[-1] if value_losses else 0
            pl = policy_losses[-1] if policy_losses else 0
            logging.info(
                f"[검증] Ep {i_ep:4d} | AvgReward: {avg_r:.4f} | "
                f"CTR: {ctr:.4f} | VLoss: {vl:.4f} | PLoss: {pl:.4f}"
            )

        # ── 체크포인트 저장 (500 에피소드마다) ───────────────────────
        if i_ep > 0 and i_ep % checkpoint_interval == 0:
            agent.save_checkpoint(
                i_ep, rewards, val_rewards, val_ctr,
                value_losses, policy_losses
            )

    # ── 최종 저장 ──────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    agent.save_model("LM_DDPG_T1_final")
    np.save("output/lm_ddpg_rewards",       np.array(rewards))
    np.save("output/lm_ddpg_val_rewards",   np.array(val_rewards))
    np.save("output/lm_ddpg_val_ctr",       np.array(val_ctr))
    np.save("output/lm_ddpg_value_losses",  np.array(value_losses))
    np.save("output/lm_ddpg_policy_losses", np.array(policy_losses))
    env.close()
    logging.info("학습 완료.")


# =======================================================================
# [섹션 7] 진입점
# =======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LM-DDPG 학습")
    parser.add_argument("--resume", action="store_true",
                        help="가장 최근 체크포인트에서 이어서 학습")
    args = parser.parse_args()
:wq    main(resume=args.resume)
