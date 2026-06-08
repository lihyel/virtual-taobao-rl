"""
=======================================================================
T2/t2_textify.py  —  T2 텍스트화 모듈 (속성 순차 추가)
=======================================================================

[파일 위치]
  your_project/
  ├── t1_textify.py          ← T1 (확정 4개 속성, 변경하지 않음)
  └── T2/
      ├── t2_textify.py      ← 이 파일 (T2 속성 순차 추가)
      └── t2_run_multiseed.py

[T2 실험의 목적]
  T1: 확정 4개 정적 속성 (browsing_category, age_level, gender, pvalue_level)
  T2: T1 + 미확정 7개 속성을 한 번에 추가하지 않고 하나씩 추가하며 성능 변화 측정

  한 번에 7개를 추가하면 어떤 속성이 성능에 기여했는지 알 수 없음.
  한 개씩 추가하면 각 속성의 독립적 기여도를 Ablation으로 분리 가능.

[Ali-CCP 속성 레이블 사용의 타당성]
  Virtual-Taobao는 보안상 88D 정적변수의 속성명을 비공개로 함.
  본 연구는 세 독립 출처 교차검증으로 인덱스 경계를 확정하고
  Ali-CCP(알리바바 Alimama 공개 데이터셋)의 스키마를 참조하여 레이블을 추정.

  ★ 중요: Ali-CCP 데이터를 State에 추가하거나 시뮬레이터를 수정한 것이 아님.
          VT 환경이 반환하는 91D 수치는 그대로 사용하고,
          수치의 의미를 해석하는 텍스트 레이블만 Ali-CCP 스키마에서 참조.
          이 추정 레이블의 영향은 T1 vs T2 비교로 실험적으로 측정.

[T2 속성 추가 순서 — 신뢰도 + 추천 직관성 기준]
  T2-Step1: is_occupied   (VT#7,  [62~63], 2개)   ← 크기 2 일치, 소비패턴 직관적
  T2-Step2: intentions    (VT#2,  [16~26], 11개)  ← 구매 의도, 추천 직접 연관
  T2-Step3: geography     (VT#3,  [27~37], 11개)  ← 지역 선호 차이
  T2-Step4: profile_type  (VT#4,  [38~48], 11개)  ← 사용자 세그먼트
  T2-Step5: user_group    (VT#5,  [49~59], 11개)  ← 집단 분류
  T2-Step6: shops         (VT#9,  [67~84], 18개)  ← multi-hot, 구매 상점
  T2-Step7: brands        (VT#10, [85~87], 3개)   ← 선호 브랜드, 가장 불확실

[사용 방법]
  from T2.t2_textify import prepare_state_T2, GroupInfoExtractor

  # Step 지정 (1~7)
  result = prepare_state_T2(state_91d, group_extractor, step=1)
  result = prepare_state_T2(state_91d, group_extractor, step=3)

  # step=0이면 T1과 동일 (기준선 재확인용)
  result = prepare_state_T2(state_91d, group_extractor, step=0)
=======================================================================
"""

import numpy as np
import pickle
import os
import sys
from typing import Optional

# T1 모듈에서 공통 요소 재사용 (GroupInfoExtractor, 동적변수 처리 등)
# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from t1_textify import (
    GroupInfoExtractor,
    get_dynamic_numeric,
    # 인덱스 상수
    IDX_CATEGORY_START, IDX_CATEGORY_END,
    IDX_AGE_START,      IDX_AGE_END,
    IDX_GENDER_START,   IDX_GENDER_END,
    IDX_POWER_START,    IDX_POWER_END,
    IDX_PREV_CLICK, IDX_LEAVE_SIGNAL, IDX_SESSION_STEP,
    DIM_DISTILBERT, DIM_DYNAMIC_RAW, DIM_ACTOR_INPUT,
    # T1 레이블 테이블
    CATEGORY_LABELS, AGE_LABELS, GENDER_LABELS, POWER_LABELS,
)

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =======================================================================
# [섹션 1] T2 추가 속성 인덱스 정의
# =======================================================================
# 근거: UserModel.py softmax_feature() 코드에서 확정된 인덱스 경계
#       속성 레이블: Ali-CCP 스키마 참조 (추정값, 실험으로 영향 측정)

# T2-Step1: VT#7 [62~63] — user_is_occupied (직업 유무)
IDX_OCCUPIED_START, IDX_OCCUPIED_END = 62, 64

# T2-Step2: VT#2 [16~26] — user_intentions (구매 의도)
IDX_INTENTIONS_START, IDX_INTENTIONS_END = 16, 27

# T2-Step3: VT#3 [27~37] — user_geography (지역/위치)
IDX_GEOGRAPHY_START, IDX_GEOGRAPHY_END = 27, 38

# T2-Step4: VT#4 [38~48] — user_profile (프로필 유형)
IDX_PROFILE_START, IDX_PROFILE_END = 38, 49

# T2-Step5: VT#5 [49~59] — user_group (그룹 분류)
IDX_GROUP_START, IDX_GROUP_END = 49, 60

# T2-Step6: VT#9 [67~84] — user_shops (구매 상점, multi-hot 가능)
IDX_SHOPS_START, IDX_SHOPS_END = 67, 85

# T2-Step7: VT#10 [85~87] — user_brands (선호 브랜드)
IDX_BRANDS_START, IDX_BRANDS_END = 85, 88


# =======================================================================
# [섹션 2] T2 추가 속성 레이블 테이블
# =======================================================================
# ★ 모두 Ali-CCP 스키마 기반 추정값
#   레이블 자체의 순서(0번=어느 값)는 추정이며,
#   T2 실험의 목적 중 하나는 이 추정이 성능에 미치는 영향을 측정하는 것임

# is_occupied: Ali-CCP user_is_occupied (0=일반, 1=대학생/직업군)
IS_OCCUPIED_LABELS = {0: "일반 직장인", 1: "대학생/특수직종"}

# intentions: Ali-CCP user_intentions (구매 의도 11개 카테고리)
# 실제 카테고리 명칭은 비공개 → 번호로 표기
INTENTIONS_LABELS = {i: f"구매의도유형{i+1}" for i in range(11)}

# geography: Ali-CCP user_geography (지역 11개)
GEOGRAPHY_LABELS = {i: f"지역유형{i+1}" for i in range(11)}

# profile_type: Ali-CCP user_profile (프로필 유형 11개)
PROFILE_LABELS = {i: f"프로필유형{i+1}" for i in range(11)}

# user_group: Ali-CCP user_group (그룹 분류 11개)
USER_GROUP_LABELS = {i: f"사용자그룹{i+1}" for i in range(11)}

# shops: Ali-CCP user_shops (multi-hot, 상점 18개)
# multi-hot이므로 argmax 대신 상위 활성화 인덱스들 사용
SHOPS_LABELS = {i: f"상점유형{i+1}" for i in range(18)}

# brands: Ali-CCP user_brands (선호 브랜드 3개)
BRANDS_LABELS = {0: "브랜드A", 1: "브랜드B", 2: "브랜드C"}


# =======================================================================
# [섹션 3] 각 Step별 속성 추출 함수
# =======================================================================

def _extract_t1_base(state_91d: np.ndarray) -> dict:
    """T1 기준 4개 정적 속성 추출 (모든 T2 Step의 공통 기반)."""
    return {
        "cat_label":    CATEGORY_LABELS.get(
            int(np.argmax(state_91d[IDX_CATEGORY_START:IDX_CATEGORY_END])), "?"),
        "age_label":    AGE_LABELS.get(
            int(np.argmax(state_91d[IDX_AGE_START:IDX_AGE_END])), "?"),
        "gender_label": GENDER_LABELS.get(
            int(np.argmax(state_91d[IDX_GENDER_START:IDX_GENDER_END])), "?"),
        "power_label":  POWER_LABELS.get(
            int(np.argmax(state_91d[IDX_POWER_START:IDX_POWER_END])), "?"),
    }


def _extract_step_attrs(state_91d: np.ndarray, step: int) -> str:
    """
    step에 따라 추가할 속성 텍스트를 반환.
    step=0: T1과 동일 (추가 속성 없음)
    step=1: is_occupied 추가
    step=2: is_occupied + intentions 추가
    ...누적 방식
    """
    extras = []

    if step >= 1:
        # T2-Step1: is_occupied
        idx = int(np.argmax(state_91d[IDX_OCCUPIED_START:IDX_OCCUPIED_END]))
        extras.append(f"직업: {IS_OCCUPIED_LABELS.get(idx, f'직업유형{idx}')}")

    if step >= 2:
        # T2-Step2: intentions (구매 의도)
        idx = int(np.argmax(state_91d[IDX_INTENTIONS_START:IDX_INTENTIONS_END]))
        extras.append(f"구매의도: {INTENTIONS_LABELS.get(idx, f'유형{idx}')}")

    if step >= 3:
        # T2-Step3: geography (지역)
        idx = int(np.argmax(state_91d[IDX_GEOGRAPHY_START:IDX_GEOGRAPHY_END]))
        extras.append(f"거주지역: {GEOGRAPHY_LABELS.get(idx, f'지역{idx}')}")

    if step >= 4:
        # T2-Step4: profile_type (프로필 유형)
        idx = int(np.argmax(state_91d[IDX_PROFILE_START:IDX_PROFILE_END]))
        extras.append(f"프로필유형: {PROFILE_LABELS.get(idx, f'유형{idx}')}")

    if step >= 5:
        # T2-Step5: user_group (그룹 분류)
        idx = int(np.argmax(state_91d[IDX_GROUP_START:IDX_GROUP_END]))
        extras.append(f"사용자그룹: {USER_GROUP_LABELS.get(idx, f'그룹{idx}')}")

    if step >= 6:
        # T2-Step6: shops (multi-hot — 활성화된 인덱스 상위 2개 사용)
        shops_vec = state_91d[IDX_SHOPS_START:IDX_SHOPS_END]
        top2 = np.argsort(shops_vec)[::-1][:2]
        shop_names = [SHOPS_LABELS.get(i, f"상점{i}") for i in top2
                      if shops_vec[i] > 0.1]
        if shop_names:
            extras.append(f"주이용상점: {', '.join(shop_names)}")

    if step >= 7:
        # T2-Step7: brands (선호 브랜드)
        idx = int(np.argmax(state_91d[IDX_BRANDS_START:IDX_BRANDS_END]))
        extras.append(f"선호브랜드: {BRANDS_LABELS.get(idx, f'브랜드{idx}')}")

    return ", ".join(extras) if extras else ""


# =======================================================================
# [섹션 4] T2 핵심 텍스트화 함수
# =======================================================================

def textify_state_T2(
    state_91d: np.ndarray,
    group_extractor: Optional[GroupInfoExtractor] = None,
    step: int = 1,
) -> str:
    """
    T2 텍스트화 함수. T1의 4개 속성에 step만큼 속성을 누적 추가.

    Args:
        state_91d       : (91,) numpy array
        group_extractor : K-Means 집단 정보 추출기 (None 가능)
        step            : 추가 속성 수 (0=T1과 동일, 1~7)

    Returns:
        DistilBERT 입력용 자연어 프롬프트 문자열

    프롬프트 구조:
        [T1 기반 개인 프로필] + [T2 추가 속성] + [집단 트렌드] + [세션 상태]
    """
    if not isinstance(state_91d, np.ndarray):
        state_91d = np.array(state_91d, dtype=float)
    if state_91d.ndim == 2:
        state_91d = state_91d[0]
    assert 0 <= step <= 7, f"step은 0~7 사이여야 합니다. 받은 값: {step}"

    # ── T1 기반 속성 ──────────────────────────────────────────────────
    base = _extract_t1_base(state_91d)
    personal_desc = (
        f"고객 프로필: {base['age_label']} {base['gender_label']}, "
        f"{base['power_label']}, "
        f"주요 탐색 카테고리: {base['cat_label']}."
    )

    # ── T2 추가 속성 ──────────────────────────────────────────────────
    extra_desc = _extract_step_attrs(state_91d, step)
    if extra_desc:
        personal_desc = personal_desc.rstrip(".") + f", {extra_desc}."

    # ── 동적변수 ──────────────────────────────────────────────────────
    prev_click   = int(state_91d[IDX_PREV_CLICK])
    leave_val    = int(state_91d[IDX_LEAVE_SIGNAL])
    session_step = int(state_91d[IDX_SESSION_STEP])

    if leave_val <= 2:
        leave_label = "이탈 위험 낮음"
    elif leave_val <= 6:
        leave_label = "이탈 위험 중간"
    else:
        leave_label = "이탈 위험 높음"

    is_cold_start = (session_step == 0)

    if is_cold_start:
        session_desc = "첫 접속 (클릭 이력 없음, Cold-start)"
    else:
        if prev_click == 0:
            click_desc = "클릭 없음"
        elif prev_click == 1:
            click_desc = f"직전 {prev_click}번 클릭"
        else:
            click_desc = f"직전 {prev_click}번 클릭 (적극 탐색)"
        session_desc = f"세션 {session_step}번째 진행 중, {click_desc}"

    # ── 집단 정보 ──────────────────────────────────────────────────────
    group_text = group_extractor.get_group_text(state_91d) \
        if group_extractor is not None else None

    # ── 프롬프트 조합 ─────────────────────────────────────────────────
    if group_text and is_cold_start:
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
# [섹션 5] prepare_state_T2 — RL 루프 통합 래퍼
# =======================================================================

def prepare_state_T2(
    state_91d: np.ndarray,
    group_extractor: Optional[GroupInfoExtractor] = None,
    step: int = 1,
    normalize_dynamic: bool = True,
) -> dict:
    """
    T2 텍스트화 래퍼. t2_run_multiseed.py의 RL 루프에서 호출.

    Args:
        state_91d       : (91,) numpy array
        group_extractor : K-Means 추출기
        step            : T2 속성 추가 단계 (0~7)
        normalize_dynamic: 동적변수 정규화 여부

    Returns:
        {
          "prompt"         : DistilBERT 입력 문자열
          "dynamic_numeric": (3,) numpy array
          "is_cold_start"  : bool
          "step"           : 현재 T2 step 번호
        }
    """
    if not isinstance(state_91d, np.ndarray):
        state_91d = np.array(state_91d, dtype=float)
    if state_91d.ndim == 2:
        state_91d = state_91d[0]

    return {
        "prompt":          textify_state_T2(state_91d, group_extractor, step),
        "dynamic_numeric": get_dynamic_numeric(state_91d, normalize_dynamic),
        "is_cold_start":   (int(state_91d[IDX_SESSION_STEP]) == 0),
        "step":            step,
    }


# =======================================================================
# [섹션 6] T2 Step 메타데이터 (실험 이름, 파일명 등에 사용)
# =======================================================================

T2_STEP_META = {
    0: {"name": "T2-Step0 (=T1 기준선)", "attr": "없음 (T1 동일)",   "folder": "step0_baseline"},
    1: {"name": "T2-Step1",              "attr": "is_occupied",       "folder": "step1_occupied"},
    2: {"name": "T2-Step2",              "attr": "+intentions",       "folder": "step2_intentions"},
    3: {"name": "T2-Step3",              "attr": "+geography",        "folder": "step3_geography"},
    4: {"name": "T2-Step4",              "attr": "+profile_type",     "folder": "step4_profile"},
    5: {"name": "T2-Step5",              "attr": "+user_group",       "folder": "step5_group"},
    6: {"name": "T2-Step6",              "attr": "+shops",            "folder": "step6_shops"},
    7: {"name": "T2-Step7",              "attr": "+brands",           "folder": "step7_brands"},
}


# =======================================================================
# [섹션 7] 검증
# =======================================================================

if __name__ == "__main__":
    print("="*65)
    print("t2_textify.py 검증 — 속성 순차 추가 확인")
    print("="*65)

    # 테스트 state 생성
    s = np.zeros(91)
    s[3]  = 1   # category=뷰티
    s[10] = 1   # age=20대초반
    s[61] = 1   # gender=여성
    s[66] = 1   # power=고소비층
    s[62] = 1   # is_occupied=대학생 (T2-Step1)
    s[20] = 1   # intentions=구매의도유형5 (T2-Step2)
    s[88] = 0   # prev_click=0
    s[89] = 1   # leave=낮음
    s[90] = 0   # step=0 (Cold-start)

    print(f"\n{'─'*60}")
    print("Cold-start 사용자, Step별 프롬프트 변화:")
    print(f"{'─'*60}")
    for step in range(8):
        meta = T2_STEP_META[step]
        prompt = textify_state_T2(s, group_extractor=None, step=step)
        print(f"\n  [{meta['name']}] 추가 속성: {meta['attr']}")
        print(f"  → \"{prompt[:100]}{'...' if len(prompt)>100 else ''}\"")

    print()
    print("="*65)
    print("실험 폴더명 규칙:")
    print("="*65)
    for step, meta in T2_STEP_META.items():
        print(f"  output/T2/{meta['folder']}/seed_{{N}}/")