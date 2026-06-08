"""
=======================================================================
textify.py  —  T1 텍스트화 모듈 (최종판)
=======================================================================

[파일 위치]
  프로젝트 루트에 배치:
  your_project/
  ├── textify.py              ← 이 파일
  ├── distilbert_encoder.py
  ├── trainer_lm_ddpg.py
  └── ...

[이 파일이 담당하는 것]
  Virtual-Taobao 91D state 벡터를 자연어 프롬프트로 변환.
  K-Means 클러스터링 결과(집단 정보)를 프롬프트에 통합하여
  단일 LLM 에이전트가 개인+집단 선호를 동시에 추론하도록 설계.

[김홍(2023) 연구와의 관계]
  김홍: MDP(P) + MDP(G) 이중 에이전트
        → 개인 state(91D) / 집단 centroid state(91D)를 각각 별도 DDPG에 입력
  본 연구: 단일 LLM 에이전트
        → 개인 정보 + 집단 트렌드를 하나의 자연어 프롬프트로 통합
        → DistilBERT가 두 정보를 동시에 인코딩하여 단일 임베딩으로 출력

[동적변수 설계 근거]
  김홍 cluster 버전(cluster_MDP_G_.py):
    State = 88D정적 + 20D(클릭이력10개×2D) + 1D(시간t) = 109D
    → VT env를 확장하여 최근 10개 클릭이력을 수치로 포함

  본 연구 (표준 VT 91D 기반):
    feat88(직전클릭수) + feat89(이탈예측) + feat90(시간t)
    → 텍스트: feat88, feat90을 LM이 이해하는 자연어 이력 요약으로 변환
              (김홍이 20D 수치로 담은 이력을 의미론적으로 더 풍부하게 표현)
    → 수치: feat88, feat89, feat90을 정규화 후 3D로 병렬 입력

[Cold-start 처리]
  김홍: "첫 시간은 동적정보가 없어 정적정보로만 추천"
  본 연구: session_step=0 감지 → "첫 접속" 명시 → LLM이 Cold-start 전략 적용
           Cold-start 시 집단 트렌드 정보를 더 강조 → 개인 이력 없을 때 보완
=======================================================================
"""

import numpy as np
import pickle
import os
from typing import Optional, Tuple

# ── 선택적 임포트: sklearn은 클러스터링에만 필요 ─────────────────────
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[textify] sklearn 없음 — 집단 정보 없이 실행됩니다.")


# =======================================================================
# [섹션 1] 인덱스 상수 정의
# =======================================================================
# 근거: UserModel.py softmax_feature() 코드에서 직접 추출 (100% 확정)

# ── 정적변수 88D 서브벡터 ───────────────────────────────────────────
IDX_CATEGORY_START, IDX_CATEGORY_END = 0,  8   # browsing_category (8개) ✅
IDX_AGE_START,      IDX_AGE_END      = 8,  16  # age_level         (8개) ✅
IDX_UNK_A_START,    IDX_UNK_A_END    = 16, 27  # 미확정 (T1 미사용)
IDX_UNK_B_START,    IDX_UNK_B_END    = 27, 38  # 미확정 (T1 미사용)
IDX_UNK_C_START,    IDX_UNK_C_END    = 38, 49  # 미확정 (T1 미사용)
IDX_UNK_D_START,    IDX_UNK_D_END    = 49, 60  # 미확정 (T1 미사용)
IDX_GENDER_START,   IDX_GENDER_END   = 60, 62  # gender            (2개) ✅
IDX_UNK_E_START,    IDX_UNK_E_END    = 62, 64  # 미확정 (T1 미사용)
IDX_POWER_START,    IDX_POWER_END    = 64, 67  # pvalue_level      (3개) ✅
IDX_UNK_F_START,    IDX_UNK_F_END    = 67, 85  # 미확정 (T1 미사용)
IDX_UNK_G_START,    IDX_UNK_G_END    = 85, 88  # 미확정 (T1 미사용)

# ── 동적변수 (VirtualTB.py state 프로퍼티로 확정) ─────────────────────
# state = cur_user(88D) + lst_action(2D) + [total_c](1D)
# lst_action[0] = ActionModel 출력 a (클릭수, max_a=11 → 0~10)
# lst_action[1] = ActionModel 출력 b (이탈예측, max_b=10 → 0~9)
# total_c       = 세션 누적 step = 김홍(2023)의 시간 t
IDX_PREV_CLICK    = 88   # 직전 클릭 수       ← 김홍 클릭이력 중 최신 1개
IDX_LEAVE_SIGNAL  = 89   # 이탈 예측값        ← VT 제공 추가 정보
IDX_SESSION_STEP  = 90   # 시간 t             ← 김홍의 t와 동일

# ── 아키텍처 차원 ──────────────────────────────────────────────────
DIM_DISTILBERT  = 768   # DistilBERT CLS hidden state 크기 (고정)
DIM_DYNAMIC_RAW = 3     # 동적변수 수치 병렬 입력 차원
DIM_ACTOR_INPUT = DIM_DISTILBERT + DIM_DYNAMIC_RAW  # 771D


# =======================================================================
# [섹션 2] 레이블 매핑 테이블
# =======================================================================

# browsing_category: VT Figure3 'Category Proportion'(8개) + README 'browsing history' + taobao 광고 category_code (크기 8 일치)
CATEGORY_LABELS = {
    0: "패션/의류",
    1: "전자/가전",
    2: "식품/식료품",
    3: "뷰티/화장품",
    4: "스포츠/레저",
    5: "홈/리빙",
    6: "도서/문화",
    7: "기타",
}

# age_level: VT README 'customer age' + Taobao 광고 age_level (크기 8 일치)
AGE_LABELS = {
    0: "10대 미만",
    1: "10대",
    2: "20대 초반",
    3: "20대 후반",
    4: "30대",
    5: "40대",
    6: "50대",
    7: "60대 이상",
}

# gender: VT README 'customer gender' + Taobao 광고 final_gender_code (크기 2 일치)
GENDER_LABELS = {0: "남성", 1: "여성"}

# pvalue_level: VT Figure3 'Power Proportion'(3레벨) + Taobao 광고 pvalue_level (크기 3 일치)
POWER_LABELS = {0: "저소비층", 1: "중소비층", 2: "고소비층"}


# =======================================================================
# [섹션 3] 집단 정보 모듈 (K-Means 기반)
# =======================================================================
# 근거: cluster_MDP_G_.py의 get_mean_state() 방식을 LM 통합으로 확장
# 김홍은 cluster centroid를 수치로 DDPG에 입력했고,
# 본 연구는 centroid를 텍스트로 변환하여 프롬프트에 포함

class GroupInfoExtractor:
    """
    K-Means 클러스터링으로 현재 고객이 속한 집단의 트렌드를 추출.

    김홍(2023) trainer_MDP_G_.py의 get_mean_state()를 확장:
      - 원본: centroid 88D 벡터를 DDPG 입력 수치로 사용
      - 본 연구: centroid의 확정 속성을 텍스트로 변환하여 LM 프롬프트에 통합

    Cold-start 기여:
      개인 클릭이력이 전혀 없어도(session_step=0) 집단 트렌드가 있으면
      LM이 '유사 고객들의 선호'를 바탕으로 추천 전략을 수립할 수 있음.
    """

    def __init__(
        self,
        kmeans_model: Optional[object] = None,
        n_clusters: int = 100,
        dataset_path: str = "./virtualTB/SupervisedLearning/dataset.txt",
        model_save_path: str = "./models/kmeans_T1.pkl",
    ):
        """
        Args:
            kmeans_model    : 이미 학습된 KMeans 객체. None이면 데이터셋에서 학습.
            n_clusters      : 클러스터 수. 김홍(2023) 기본값 100.
            dataset_path    : VT SupervisedLearning/dataset.txt 경로
            model_save_path : 학습된 모델 저장 경로
        """
        self.n_clusters = n_clusters
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path
        self.kmeans = kmeans_model

    def fit(self) -> None:
        """
        dataset.txt에서 88D 정적변수를 읽어 K-Means 학습.

        근거: cluster_MDP_G_.py
          dataset = np.array([np.array(row.split(","))[:88]
                              for row in pd.read_csv(dataset_path,...)[0].values])
          kmeans = KMeans(n_clusters=100, random_state=42).fit(dataset)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn이 필요합니다: pip install scikit-learn")

        if os.path.exists(self.model_save_path):
            with open(self.model_save_path, "rb") as f:
                self.kmeans = pickle.load(f)
            print(f"[GroupInfoExtractor] 저장된 K-Means 로드: {self.model_save_path}")
            return

        import pandas as pd
        print(f"[GroupInfoExtractor] K-Means 학습 중 (n_clusters={self.n_clusters})...")
        raw = pd.read_csv(self.dataset_path, delimiter="\t", header=None)[0].values
        # 각 행의 첫 88개 값 = 정적변수 (trainer_MDP_G_.py의 파싱 방식과 동일)
        dataset = np.array(
            [np.array(row.split(","))[:88] for row in raw], dtype=float
        )
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.kmeans.fit(dataset)

        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        with open(self.model_save_path, "wb") as f:
            pickle.dump(self.kmeans, f)
        print(f"[GroupInfoExtractor] 학습 완료, 저장: {self.model_save_path}")

    def get_cluster_centroid(self, state_91d: np.ndarray) -> np.ndarray:
        """
        현재 사용자의 클러스터 centroid 반환 (88D).

        cluster_MDP_G_.py의 get_mean_state()와 동일한 로직:
          kmeans.cluster_centers_[kmeans.predict([state[:88]])[0]]
        """
        assert self.kmeans is not None, "fit()을 먼저 실행하세요."
        static_88d = state_91d[:88].reshape(1, -1)
        cluster_idx = self.kmeans.predict(static_88d)[0]
        return self.kmeans.cluster_centers_[cluster_idx]

    def get_group_text(self, state_91d: np.ndarray) -> str:
        """
        현재 사용자가 속한 클러스터의 트렌드를 자연어로 반환.

        김홍(2023)은 centroid를 수치로만 사용했으나,
        본 연구는 centroid의 확정 속성을 텍스트로 변환하여
        LM이 집단 트렌드를 의미론적으로 이해하도록 함.

        Cold-start 강화 기여:
          개인 이력이 없을 때(session_step=0) 이 집단 정보가
          LLM의 주요 추론 근거로 활용됨.
        """
        if self.kmeans is None:
            return "집단 정보 없음"

        centroid = self.get_cluster_centroid(state_91d)

        # centroid에서 확정 속성만 추출 (T1 기준)
        cat_idx    = int(np.argmax(centroid[IDX_CATEGORY_START:IDX_CATEGORY_END]))
        age_idx    = int(np.argmax(centroid[IDX_AGE_START:IDX_AGE_END]))
        gender_idx = int(np.argmax(centroid[IDX_GENDER_START:IDX_GENDER_END]))
        power_idx  = int(np.argmax(centroid[IDX_POWER_START:IDX_POWER_END]))

        cat_label    = CATEGORY_LABELS.get(cat_idx, f"카테고리{cat_idx}")
        age_label    = AGE_LABELS.get(age_idx, f"연령대{age_idx}")
        gender_label = GENDER_LABELS.get(gender_idx, f"성별{gender_idx}")
        power_label  = POWER_LABELS.get(power_idx, f"소비레벨{power_idx}")

        return (
            f"유사 고객군 트렌드: {age_label} {gender_label} {power_label} 고객들이 "
            f"{cat_label} 카테고리를 주로 탐색"
        )


# =======================================================================
# [섹션 4] 핵심 텍스트화 함수
# =======================================================================

def textify_state_T1(
    state_91d: np.ndarray,
    group_extractor: Optional[GroupInfoExtractor] = None,
) -> str:
    """
    91D 상태 벡터 → 자연어 프롬프트 (T1 조건).

    [개인 정보] 정적 4개 속성 → 자연어 페르소나
    [집단 정보] K-Means centroid → 유사 고객 트렌드 (group_extractor 있을 때)
    [동적 정보] feat88(클릭), feat89(이탈), feat90(t) → 세션 상태 서술

    Cold-start 처리:
      session_step=0 감지 → "첫 접속" 명시 + 집단 트렌드 강조
      김홍(2023): "첫 시간은 동적정보가 없어 정적정보로만 추천"
      본 연구: 정적 + 집단 트렌드로 LM 추론 → Cold-start 더 풍부하게 해결

    Args:
        state_91d       : (91,) numpy array, env.step()/env.reset() 반환값
        group_extractor : GroupInfoExtractor 인스턴스 (None이면 집단 정보 생략)

    Returns:
        DistilBERT 입력용 자연어 프롬프트 문자열
    """
    if not isinstance(state_91d, np.ndarray):
        state_91d = np.array(state_91d, dtype=float)
    if state_91d.ndim == 2:
        state_91d = state_91d[0]
    assert state_91d.shape[-1] == 91, f"91D 필요, 받은 크기: {state_91d.shape}"

    # ── [1] 개인 정적변수 추출 ────────────────────────────────────────
    # argmax(): one-hot 벡터에서 활성화된 카테고리 인덱스 추출
    # GAN-SD 생성값도 softmax 확률이므로 argmax = 최고 확률 카테고리
    cat_idx    = int(np.argmax(state_91d[IDX_CATEGORY_START:IDX_CATEGORY_END]))
    age_idx    = int(np.argmax(state_91d[IDX_AGE_START:IDX_AGE_END]))
    gender_idx = int(np.argmax(state_91d[IDX_GENDER_START:IDX_GENDER_END]))
    power_idx  = int(np.argmax(state_91d[IDX_POWER_START:IDX_POWER_END]))

    cat_label    = CATEGORY_LABELS.get(cat_idx,    f"카테고리{cat_idx}")
    age_label    = AGE_LABELS.get(age_idx,         f"연령대{age_idx}")
    gender_label = GENDER_LABELS.get(gender_idx,   f"성별{gender_idx}")
    power_label  = POWER_LABELS.get(power_idx,     f"소비레벨{power_idx}")

    # ── [2] 동적변수 추출 ─────────────────────────────────────────────
    # feat88: 직전 클릭수 ← 김홍의 최근 클릭이력 중 가장 최신 1개에 대응
    prev_click   = int(state_91d[IDX_PREV_CLICK])
    # feat89: 이탈 예측값 (ActionModel 출력 b, 0~9)
    leave_val    = int(state_91d[IDX_LEAVE_SIGNAL])
    # feat90: 시간 t = 김홍(2023)의 t ∈ {0, 1, 2, ...}
    session_step = int(state_91d[IDX_SESSION_STEP])

    # 이탈 위험도 3단계 해석
    if leave_val <= 2:
        leave_label = "이탈 위험 낮음"
    elif leave_val <= 6:
        leave_label = "이탈 위험 중간"
    else:
        leave_label = "이탈 위험 높음"

    # ── [3] Cold-start 판단 및 세션 서술 ──────────────────────────────
    # 김홍(2023): "클릭을 한 번도 하지 않는 경우 동적변수도 변동이 없고
    #              클릭을 해야 St+1하게 된다" → step=0이 Cold-start
    is_cold_start = (session_step == 0)

    if is_cold_start:
        # Cold-start: 개인 클릭이력 전무 → 집단 트렌드가 핵심 추론 근거
        session_desc = "첫 접속 (클릭 이력 없음, Cold-start)"
    else:
        # 세션 진행 중
        if prev_click == 0:
            click_desc = "클릭 없음"
        elif prev_click == 1:
            click_desc = f"직전 {prev_click}번 클릭"
        else:
            click_desc = f"직전 {prev_click}번 클릭 (적극 탐색)"
        # 김홍의 클릭이력 t: "10개의 범위는 [ti-10, ti-1]"
        # 본 연구는 가장 최근 1 step 요약으로 표현
        session_desc = (
            f"세션 {session_step}번째 진행 중 "
            f"(김홍 t={session_step}), {click_desc}"
        )

    # ── [4] 집단 정보 추가 ────────────────────────────────────────────
    # group_extractor가 있으면 K-Means centroid를 텍스트로 변환
    # Cold-start 시 집단 트렌드를 더 앞쪽에 배치하여 LM 추론 강화
    if group_extractor is not None:
        group_text = group_extractor.get_group_text(state_91d)
    else:
        group_text = None

    # ── [5] 프롬프트 조합 ─────────────────────────────────────────────
    # 구조: [개인 프로필] → [집단 트렌드] → [현재 세션] → [이탈 신호]
    # Cold-start 시: 집단 트렌드를 더 명시적으로 강조

    personal_desc = (
        f"고객 프로필: {age_label} {gender_label}, {power_label}, "
        f"주요 탐색 카테고리: {cat_label}."
    )

    if group_text and is_cold_start:
        # Cold-start: 집단 정보를 더 앞에 배치하여 LM이 더 많이 참조하도록
        prompt = (
            f"{personal_desc} "
            f"[Cold-start: 개인 이력 없음] {group_text}. "
            f"현재 세션: {session_desc}, {leave_label}."
        )
    elif group_text:
        prompt = (
            f"{personal_desc} "
            f"{group_text}. "
            f"현재 세션: {session_desc}, {leave_label}."
        )
    else:
        prompt = (
            f"{personal_desc} "
            f"현재 세션: {session_desc}, {leave_label}."
        )

    return prompt


# =======================================================================
# [섹션 5] 동적변수 수치 추출 (병렬 입력용)
# =======================================================================

def get_dynamic_numeric(
    state_91d: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    동적변수 3개를 정규화된 수치 배열로 반환.

    텍스트화에서 놓칠 수 있는 정밀한 수치 정보를 보존하기 위해
    DistilBERT 임베딩과 concat하여 Actor에 입력.

    정규화 기준:
      feat88 / 10.0  : ActionModel max_a=11 → 실질 최대값 10
      feat89 /  9.0  : ActionModel max_b=10 → 0-indexed 최대값 9
      feat90 / 100.0 : VirtualTB max_c=100

    Returns:
        shape (3,) numpy array
    """
    if not isinstance(state_91d, np.ndarray):
        state_91d = np.array(state_91d, dtype=float)
    if state_91d.ndim == 2:
        state_91d = state_91d[0]

    prev_click   = float(state_91d[IDX_PREV_CLICK])
    leave_signal = float(state_91d[IDX_LEAVE_SIGNAL])
    session_step = float(state_91d[IDX_SESSION_STEP])

    if normalize:
        prev_click   /= 10.0
        leave_signal /=  9.0
        session_step /= 100.0

    return np.array([prev_click, leave_signal, session_step], dtype=np.float32)


# =======================================================================
# [섹션 6] 전체 파이프라인 래퍼
# =======================================================================

def prepare_state(
    state_91d: np.ndarray,
    group_extractor: Optional[GroupInfoExtractor] = None,
    normalize_dynamic: bool = True,
) -> dict:
    """
    env.step()/env.reset() 반환값을 받아
    프롬프트와 동적수치를 모두 포함한 딕셔너리 반환.

    RL 루프에서 매 step마다 호출.

    Returns:
        {
          "prompt"        : DistilBERT 입력 문자열
          "dynamic_numeric": (3,) numpy array (정규화 완료)
          "is_cold_start" : bool
          "parsed"        : 추출된 속성값 딕셔너리 (디버깅용)
        }
    """
    if not isinstance(state_91d, np.ndarray):
        state_91d = np.array(state_91d, dtype=float)
    if state_91d.ndim == 2:
        state_91d = state_91d[0]

    session_step = int(state_91d[IDX_SESSION_STEP])

    return {
        "prompt": textify_state_T1(state_91d, group_extractor),
        "dynamic_numeric": get_dynamic_numeric(state_91d, normalize_dynamic),
        "is_cold_start": (session_step == 0),
        "parsed": {
            "browsing_category": CATEGORY_LABELS.get(
                int(np.argmax(state_91d[IDX_CATEGORY_START:IDX_CATEGORY_END])), "?"),
            "age":    AGE_LABELS.get(
                int(np.argmax(state_91d[IDX_AGE_START:IDX_AGE_END])), "?"),
            "gender": GENDER_LABELS.get(
                int(np.argmax(state_91d[IDX_GENDER_START:IDX_GENDER_END])), "?"),
            "power":  POWER_LABELS.get(
                int(np.argmax(state_91d[IDX_POWER_START:IDX_POWER_END])), "?"),
            "prev_click":   int(state_91d[IDX_PREV_CLICK]),
            "leave_signal": int(state_91d[IDX_LEAVE_SIGNAL]),
            "session_step": session_step,
        },
    }


# =======================================================================
# [섹션 7] 검증
# =======================================================================

if __name__ == "__main__":

    print("="*65)
    print("textify.py 검증")
    print("="*65)

    def make_state(cat=0, age=2, gender=1, power=2,
                   click=0, leave=1, step=0):
        s = np.zeros(91)
        s[IDX_CATEGORY_START + cat]  = 1
        s[IDX_AGE_START      + age]  = 1
        s[IDX_GENDER_START   + gender] = 1
        s[IDX_POWER_START    + power]  = 1
        s[IDX_PREV_CLICK]    = click
        s[IDX_LEAVE_SIGNAL]  = leave
        s[IDX_SESSION_STEP]  = step
        return s

    cases = [
        ("Cold-start: 20대초반 여성, 고소비층, 뷰티",
         make_state(cat=3, age=2, gender=1, power=2, click=0, leave=1, step=0)),
        ("세션 진행 중: 이탈 위험 높음, 클릭 5회, t=12",
         make_state(cat=3, age=2, gender=1, power=2, click=5, leave=8, step=12)),
        ("Cold-start: 40대 남성, 저소비층, 전자/가전",
         make_state(cat=1, age=4, gender=0, power=0, click=0, leave=0, step=0)),
        ("진행 중: 30대 여성, 중소비층, 홈/리빙, t=7",
         make_state(cat=5, age=4, gender=1, power=1, click=2, leave=4, step=7)),
    ]

    # 집단 정보 없이 (group_extractor=None)
    print("\n[집단 정보 없이 실행 (T1 기본)]")
    for label, state in cases:
        result = prepare_state(state, group_extractor=None)
        print(f"\n  [{label}]")
        print(f"  Cold-start: {result['is_cold_start']}")
        print(f"  동적수치:   {result['dynamic_numeric']}")
        print(f"  프롬프트: \"{result['prompt']}\"")

    print()
    print("="*65)
    print("아키텍처 정보")
    print("="*65)
    print(f"  DistilBERT 출력 차원: {DIM_DISTILBERT}D")
    print(f"  동적변수 수치 차원:   {DIM_DYNAMIC_RAW}D")
    print(f"  Actor 입력 차원:     {DIM_ACTOR_INPUT}D")
    print(f"  Actor 출력 차원:     27D (VT item weight)")
    print()
    print("  김홍(2023) 대비 동적변수 처리:")
    print("    김홍: 클릭이력 20D(수치) + 시간t 1D = 21D → DDPG 수치 입력")
    print("    본연구: feat88,89,90 텍스트(LM 이해) + 수치 3D 병렬")
    print("            → LM이 클릭이력을 의미론적으로 이해하여 추론")
