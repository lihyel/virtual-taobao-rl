"""
=======================================================================
distilbert_encoder.py  —  DistilBERT 인코딩 래퍼
=======================================================================

[파일 위치]
  your_project/
  ├── textify.py
  ├── distilbert_encoder.py   ← 이 파일
  ├── lm_ddpg.py
  ├── trainer_lm_ddpg.py
  └── ...

[역할]
  textify.py가 생성한 자연어 프롬프트를 DistilBERT로 인코딩하여
  768D 임베딩 벡터를 반환.
  이후 동적변수 수치(3D)와 concat하여 DDPG Actor의 입력(771D)을 생성.

[Wu et al.(2024) 방법론 구현]
  "We have innovated by substituting the traditional MLPs with a
   distilled and pre-trained BERT model serving as the RL agent.
   This model is utilized to encode the concatenated sentences,
   which represent the state variables, into feature embeddings."

[아키텍처]
  프롬프트 → DistilBERT → [CLS] 토큰 hidden state → 768D 임베딩
  동적수치  ──────────────────────────────────────────────→ 3D
  concat                                                 → 771D
                                                           ↓
                                                     DDPG Actor
                                                           ↓
                                                      27D weight
=======================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union


# =======================================================================
# [섹션 1] DistilBERT 인코더
# =======================================================================

class DistilBERTEncoder(nn.Module):
    """
    프롬프트 문자열 → 768D 임베딩 변환기.

    Wu et al.(2024)이 농업 상태 변수에 적용한 방식과 동일한 원리:
    - 자연어 상태를 DistilBERT의 [CLS] 토큰 출력으로 인코딩
    - RL 학습 중 fine-tuning 가능 (trainable=True 시)

    Args:
        model_name  : HuggingFace 모델 이름 (기본: distilbert-base-uncased)
        trainable   : DistilBERT 파라미터를 RL 학습 중 업데이트할지 여부
                      Wu et al.(2024)은 fine-tuning 방식 사용.
                      컴퓨팅 자원이 제한적이면 False로 고정 (frozen encoder).
        device      : "cuda" 또는 "cpu"
        max_length  : 최대 토큰 길이 (기본 128로 충분, 프롬프트가 짧으므로)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        trainable: bool = True,
        device: str = "cpu",
        max_length: int = 128,
    ):
        super().__init__()
        self.device = device
        self.max_length = max_length
        self.trainable = trainable

        # transformers 라이브러리 로드
        try:
            from transformers import DistilBertTokenizer, DistilBertModel
        except ImportError:
            raise ImportError(
                "transformers 패키지가 필요합니다:\n"
                "  pip install transformers"
            )

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.bert = self.bert.to(device)

        if not trainable:
            # frozen encoder: DistilBERT 파라미터 고정
            # 컴퓨팅 자원 제한 시 사용. M1 Pro 환경에서 권장.
            for param in self.bert.parameters():
                param.requires_grad = False

        self.output_dim = 768  # DistilBERT hidden size (고정값)

    def encode(self, prompt: Union[str, List[str]]) -> torch.Tensor:
        """
        프롬프트 → 768D 임베딩.

        [CLS] 토큰을 문장 전체의 의미 표현으로 사용.
        Wu et al.(2024): "DistilBERT encoding the concatenated sentences
                          into feature embeddings"

        Args:
            prompt: 단일 문자열 또는 문자열 리스트 (배치 처리)

        Returns:
            shape (batch_size, 768) 또는 (768,) torch.Tensor
        """
        single = isinstance(prompt, str)
        if single:
            prompt = [prompt]

        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # DistilBERT 순전파
        # trainable=True: grad 추적 (RL 학습 중 fine-tuning)
        # trainable=False: no_grad (추론만)
        if self.trainable:
            outputs = self.bert(**inputs)
        else:
            with torch.no_grad():
                outputs = self.bert(**inputs)

        # [CLS] 토큰 (인덱스 0)의 hidden state → 문장 전체 의미 벡터
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        if single:
            return cls_embedding.squeeze(0)  # (768,)
        return cls_embedding  # (batch, 768)

    def forward(self, prompt: Union[str, List[str]]) -> torch.Tensor:
        """nn.Module 인터페이스 호환. encode()와 동일."""
        return self.encode(prompt)


# =======================================================================
# [섹션 2] 하이브리드 State 인코더
# =======================================================================

class HybridStateEncoder(nn.Module):
    """
    텍스트 임베딩(768D) + 동적수치(3D) → concat → 771D.

    이 클래스가 DDPG Actor의 실질적 입력 전처리기.

    [설계 근거]
    동적변수를 텍스트로만 표현하면 이산 정수값(클릭수 등)의
    정밀도가 손실될 수 있음 → 수치로도 병렬 입력.
    Wu et al.(2024)의 구조를 마케팅 도메인에 맞게 확장.

    김홍(2023) 대응:
      김홍: 91D 수치(개인) + 91D 수치(집단 centroid) → 각각 별도 DDPG
      본 연구: 텍스트(개인+집단 통합) → DistilBERT(768D) + 수치(3D) → 단일 DDPG
    """

    def __init__(self, bert_encoder: DistilBERTEncoder):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.output_dim = bert_encoder.output_dim + 3  # 768 + 3 = 771

    def forward(
        self,
        prompt: Union[str, List[str]],
        dynamic_numeric: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt         : 텍스트화 함수 출력 문자열 (또는 리스트)
            dynamic_numeric: (batch, 3) 또는 (3,) 정규화된 동적변수 수치

        Returns:
            (batch, 771) 또는 (771,) Actor 입력 텐서
        """
        # DistilBERT 인코딩
        text_emb = self.bert_encoder.encode(prompt)  # (batch, 768) or (768,)

        # 차원 맞추기
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)  # (1, 768)
        if dynamic_numeric.dim() == 1:
            dynamic_numeric = dynamic_numeric.unsqueeze(0)  # (1, 3)

        # concat → (batch, 771)
        combined = torch.cat([text_emb, dynamic_numeric], dim=-1)

        if combined.shape[0] == 1:
            return combined.squeeze(0)  # (771,) 단일 샘플
        return combined  # (batch, 771)


# =======================================================================
# [섹션 3] DDPG와 연동하는 LM 기반 Actor/Critic
# =======================================================================

class LMActor(nn.Module):
    """
    Wu et al.(2024) 방식의 LM 기반 Actor.

    원본 ddpg.py의 Actor를 그대로 유지하되,
    입력 전처리를 HybridStateEncoder로 교체.

    원본 Actor (ddpg.py):
      linear1: num_inputs(91) → hidden_size(128)

    LMActor:
      HybridStateEncoder: prompt + dynamic_numeric → 771D
      linear1: 771 → hidden_size(128)
      (이후 구조 동일)
    """

    def __init__(
        self,
        hidden_size: int,
        action_space,
        bert_encoder: DistilBERTEncoder,
    ):
        super().__init__()
        self.hybrid_encoder = HybridStateEncoder(bert_encoder)
        num_outputs = action_space.shape[0]  # 27

        # ddpg.py Actor와 동일한 구조, 입력 차원만 771로 변경
        self.linear1 = nn.Linear(self.hybrid_encoder.output_dim, hidden_size)
        self.ln1     = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2     = nn.LayerNorm(hidden_size)
        self.mu      = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(
        self,
        prompt: Union[str, List[str]],
        dynamic_numeric: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prompt         : textify.py가 생성한 자연어 프롬프트
            dynamic_numeric: 정규화된 동적변수 수치 (3D)

        Returns:
            (27,) item weight 벡터 (tanh → [-1, 1] 범위)
        """
        x = self.hybrid_encoder(prompt, dynamic_numeric)
        x = torch.relu(self.ln1(self.linear1(x)))
        x = torch.relu(self.ln2(self.linear2(x)))
        return torch.tanh(self.mu(x))


class LMCritic(nn.Module):
    """
    ddpg.py Critic과 동일한 구조, 입력 차원을 771로 변경.

    Q(s, a) = Critic(state_emb, action)
      state_emb: 771D (DistilBERT 임베딩 + 동적수치)
      action:     27D
    """

    def __init__(
        self,
        hidden_size: int,
        action_space,
        bert_encoder: DistilBERTEncoder,
    ):
        super().__init__()
        self.hybrid_encoder = HybridStateEncoder(bert_encoder)
        num_outputs = action_space.shape[0]  # 27

        self.linear1 = nn.Linear(self.hybrid_encoder.output_dim, hidden_size)
        self.ln1     = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size + num_outputs, hidden_size)
        self.ln2     = nn.LayerNorm(hidden_size)
        self.V       = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(
        self,
        prompt: Union[str, List[str]],
        dynamic_numeric: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        x = self.hybrid_encoder(prompt, dynamic_numeric)
        x = torch.relu(self.ln1(self.linear1(x)))
        x = torch.cat([x, actions], dim=-1)
        x = torch.relu(self.ln2(self.linear2(x)))
        return self.V(x)


# =======================================================================
# [섹션 4] 배치 처리 유틸리티
# =======================================================================

def collate_batch(
    prompts: List[str],
    dynamic_numerics: List[np.ndarray],
    device: str = "cpu",
) -> tuple:
    """
    ReplayMemory에서 샘플링된 배치를 인코더 입력 형식으로 변환.

    trainer_lm_ddpg.py의 update_parameters()에서 호출.

    Args:
        prompts         : 배치 크기만큼의 프롬프트 리스트
        dynamic_numerics: 배치 크기만큼의 (3,) numpy array 리스트
        device          : "cuda" 또는 "cpu"

    Returns:
        (prompts_list, dynamic_tensor)
        prompts_list  : List[str] — DistilBERT 배치 입력
        dynamic_tensor: (batch, 3) torch.Tensor
    """
    dynamic_tensor = torch.tensor(
        np.stack(dynamic_numerics, axis=0),
        dtype=torch.float32,
        device=device,
    )
    return prompts, dynamic_tensor


# =======================================================================
# [섹션 5] 검증 (transformers 없이 동작 확인)
# =======================================================================

if __name__ == "__main__":

    print("="*65)
    print("distilbert_encoder.py 구조 검증 (transformers 없이)")
    print("="*65)
    print()
    print("[아키텍처 차원 구조]")
    print(f"  입력: 자연어 프롬프트 → DistilBERT → CLS → 768D")
    print(f"  입력: 동적변수 수치 [feat88/10, feat89/9, feat90/100] → 3D")
    print(f"  concat → 771D → Actor")
    print()
    print("[LMActor 레이어 구조]")
    print(f"  HybridStateEncoder : prompt + dyn → 771D")
    print(f"  Linear(771 → 128)  + LayerNorm + ReLU")
    print(f"  Linear(128 → 128)  + LayerNorm + ReLU")
    print(f"  Linear(128 → 27)   + Tanh")
    print(f"  출력: (27,) item weight ∈ [-1, 1]")
    print()
    print("[LMCritic 레이어 구조]")
    print(f"  HybridStateEncoder : prompt + dyn → 771D")
    print(f"  Linear(771 → 128)  + LayerNorm + ReLU")
    print(f"  concat(128 + 27)   → 155D")
    print(f"  Linear(155 → 128)  + LayerNorm + ReLU")
    print(f"  Linear(128 → 1)    : Q값 출력")
    print()
    print("[ddpg.py와의 차이]")
    print(f"  원본 Actor: Linear({91} → 128) ← num_inputs=91")
    print(f"  LMActor:   Linear({768+3} → 128) ← DistilBERT(768) + 동적수치(3)")
    print()
    print("[trainer에서 사용 방법 예시]")
    print("""
  # ① 인코더 초기화
  encoder = DistilBERTEncoder(trainable=True, device=device)

  # ② LM-DDPG 에이전트 초기화
  actor  = LMActor(hidden_size=128, action_space=env.action_space, bert_encoder=encoder)
  critic = LMCritic(hidden_size=128, action_space=env.action_space, bert_encoder=encoder)

  # ③ RL 루프에서 state 처리
  from textify import prepare_state, GroupInfoExtractor
  state_raw = env.reset()                          # (91,) numpy
  result    = prepare_state(state_raw, group_extractor)
  prompt    = result["prompt"]                     # str
  dyn       = torch.tensor(result["dynamic_numeric"])  # (3,)

  # ④ Action 선택
  action = actor(prompt, dyn)                      # (27,) tensor

  # ⑤ Critic Q값 계산
  q_value = critic(prompt, dyn, action)            # (1,) tensor
    """)
    print()
    print("[Cold-start 시나리오 프롬프트 예시]")
    examples = [
        "고객 프로필: 20대 초반 여성, 고소비층, 주요 탐색 카테고리: 뷰티/화장품. "
        "[Cold-start: 개인 이력 없음] 유사 고객군 트렌드: 20대 초반 여성 고소비층 고객들이 "
        "뷰티/화장품 카테고리를 주로 탐색. 현재 세션: 첫 접속 (클릭 이력 없음, Cold-start), 이탈 위험 낮음.",

        "고객 프로필: 40대 남성, 저소비층, 주요 탐색 카테고리: 전자/가전. "
        "유사 고객군 트렌드: 40대 남성 저소비층 고객들이 전자/가전 카테고리를 주로 탐색. "
        "현재 세션: 세션 7번째 진행 중 (김홍 t=7), 직전 2번 클릭 (적극 탐색), 이탈 위험 중간.",
    ]
    for i, ex in enumerate(examples, 1):
        print(f"  예시{i}: \"{ex[:80]}...\"")
