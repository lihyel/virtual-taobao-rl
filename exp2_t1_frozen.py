"""
실험 2: T1 + DistilBERT Frozen
trainer_lm_ddpg_v2.py에서 trainable=False로만 변경.
목적: fine-tuning vs frozen의 성능 차이 측정.
실행: python exp2_t1_frozen.py
"""
import sys
sys.path.insert(0, '.')

# trainer_lm_ddpg_v2의 main을 재사용, encoder만 frozen으로 교체
from trainer_lm_ddpg import (
    ReplayMemory, OUNoise, LMDDPG, get_device,
    Transition  # 없으면 Transition 사용
)
import os, random, logging, numpy as np, torch
import gym
import virtualTB
from collections import namedtuple
from textify import prepare_state, GroupInfoExtractor
from distilbert_encoder import DistilBERTEncoder

Transition = namedtuple('Transition',
    ('state_91d', 'action', 'mask', 'next_state_91d', 'reward'))

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler(),
                  logging.FileHandler("training_exp2_frozen.log")]
    )

    device = get_device()
    env = gym.make('VirtualTB-v0')
    env.seed(1); np.random.seed(1); torch.manual_seed(1)  # seed 다르게

    group_extractor = GroupInfoExtractor(
        dataset_path="./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path="./models/kmeans_T1.pkl",  # 실험1과 동일 K-Means 재사용
    )
    group_extractor.fit()

    # ★ 핵심 차이: trainable=False (Frozen)
    encoder = DistilBERTEncoder(trainable=False, device=device, max_length=128)
    logging.info("실험 2: DistilBERT FROZEN (FC 레이어만 학습)")

    agent = LMDDPG(
        gamma=0.7, tau=0.003, hidden_size=128,
        action_space=env.action_space,
        encoder=encoder, device=device,
    )

    memory = ReplayMemory(10000)
    ounoise = OUNoise(env.action_space.shape[0])
    batch_size = 32
    n_episodes = 2000
    checkpoint_interval = 500

    rewards, val_rewards, val_ctr = [], [], []
    value_losses, policy_losses = [], []

    for i_ep in range(n_episodes):
        state_91d = env.reset()
        ounoise.reset()
        episode_reward = 0

        while True:
            res = prepare_state(state_91d, group_extractor)
            prompt = res['prompt']
            dyn = torch.FloatTensor(res['dynamic_numeric']).unsqueeze(0).to(device)

            action_t = agent.select_action(prompt, dyn)
            noise = torch.FloatTensor(ounoise.noise()).to(device)
            action_t = (action_t + noise).clamp(-1, 1)

            next_state_91d, reward, done, _ = env.step(action_t.cpu().numpy().squeeze())

            memory.push(
                state_91d,
                action_t.detach().cpu().unsqueeze(0),
                torch.Tensor([not done]),
                next_state_91d,
                torch.Tensor([reward]),
            )
            state_91d = next_state_91d
            episode_reward += reward

            if len(memory) > batch_size:
                for _ in range(5):
                    transitions = memory.sample(batch_size)
                    batch = Transition(*zip(*transitions))
                    v_loss, p_loss = agent.update_parameters(batch, group_extractor)
                    value_losses.append(v_loss)
                    policy_losses.append(p_loss)
            if done:
                break

        rewards.append(episode_reward)
        logging.info(f"[EXP2-Frozen] Episode {i_ep:4d} | Reward: {episode_reward:.1f}")

        if i_ep > 0 and i_ep % 20 == 0:
            val_r, val_s = 0, 0
            for _ in range(50):
                s = env.reset()
                r_ep = prepare_state(s, group_extractor)
                p = r_ep['prompt']
                d = torch.FloatTensor(r_ep['dynamic_numeric']).unsqueeze(0).to(device)
                while True:
                    a = agent.select_action(p, d).cpu().numpy().squeeze()
                    ns, r, done, _ = env.step(a)
                    val_r += r; val_s += 1
                    nr = prepare_state(ns, group_extractor)
                    p = nr['prompt']
                    d = torch.FloatTensor(nr['dynamic_numeric']).unsqueeze(0).to(device)
                    if done: break
            avg_r = val_r / 50
            ctr = val_r / val_s / 10 if val_s > 0 else 0
            val_rewards.append(avg_r); val_ctr.append(ctr)
            vl = value_losses[-1] if value_losses else 0
            pl = policy_losses[-1] if policy_losses else 0
            logging.info(f"[EXP2-검증] Ep {i_ep:4d} | AvgReward: {avg_r:.4f} | CTR: {ctr:.4f} | VLoss: {vl:.4f} | PLoss: {pl:.4f}")

        if i_ep > 0 and i_ep % checkpoint_interval == 0:
            agent.save_checkpoint(i_ep, rewards, val_rewards, val_ctr,
                                  value_losses, policy_losses)

    os.makedirs("output", exist_ok=True)
    agent.save_model("LM_DDPG_T1_Frozen")
    np.save("output/exp2_frozen_rewards", np.array(rewards))
    np.save("output/exp2_frozen_val_rewards", np.array(val_rewards))
    np.save("output/exp2_frozen_val_ctr", np.array(val_ctr))
    env.close()
    logging.info("[EXP2] 학습 완료.")

if __name__ == "__main__":
    main()
