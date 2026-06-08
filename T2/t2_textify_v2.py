"""
=======================================================================
T2/t2_textify.py  —  T2 텍스트화 모듈 (수정판 v2)
=======================================================================

[v1 → v2 수정 내역]

  문제 1 ★★★ (핵심): 레이블이 "구매의도유형5" 같은 번호 조어로만 구성
    DistilBERT 사전학습 어휘에 없는 조어 → 의미 추출 불가 → 노이즈
    수정: 모든 레이블을 의미 있는 자연어로 교체

  문제 2 ★★: shops 임계값 0.1이 너무 높아 항상 공백("")이 될 가능성
    수정: 임계값 0.1 → 0.01, top-1만 사용 (더 안정적)

  문제 3 ★: 프롬프트 구조가 단순 나열로 파괴됨 (점점 길어짐)
    수정: 속성을 개별 문장으로 분리하여 구조적 서술

  문제 4: import 의존성 정리
    T1에서 가져오는 항목을 명확히 최소화

[레이블 출처 명시]
  T1 확정 4개: UserModel.py 코드 + VT Figure3 + Ali-CCP 교차검증
  T2 추가 7개: Ali-CCP user_intentions/geography/profile/group/shops/brands
              스키마 참조 추정값. 논문에서 "합리적 추정" 명시 필요.
              T1 vs T2 비교로 레이블 추정의 영향 자체를 실험적으로 측정.

[사용 방법]
  from t2_textify import prepare_state_T2, GroupInfoExtractor

  result = prepare_state_T2(state_91d, group_extractor, step=1)
  prompt = result["prompt"]
=======================================================================
"""

import numpy as np
import os
import sys
from typing import Optional

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# T1에서 가져오는 항목을 최소화 (공통 유틸리티만)
from t1_textify import (
    GroupInfoExtractor,
    get_dynamic_numeric,
    IDX_PREV_CLICK,
    IDX_LEAVE_SIGNAL,
    IDX_SESSION_STEP,
)


# =======================================================================
# [섹션 1] T1 확정 속성 인덱스 (t1_textify.py와 동일)
# =======================================================================

IDX_CATEGORY_START, IDX_CATEGORY_END = 0,  8
IDX_AGE_START,      IDX_AGE_END       = 8,  16
IDX_GENDER_START,   IDX_GENDER_END    = 60, 62
IDX_POWER_START,    IDX_POWER_END     = 64, 67

# T1 확정 레이블 (t1_textify.py와 동일 — 중복 정의로 import 의존성 최소화)
CATEGORY_LABELS = {
    0: "패션/의류", 1: "전자/가전", 2: "식품/식료품",
    3: "뷰티/화장품", 4: "스포츠/레저", 5: "홈/리빙",
    6: "도서/문화", 7: "기타",
}
AGE_LABELS = {
    0: "10대 미만", 1: "10대", 2: "20대 초반", 3: "20대 후반",
    4: "30대", 5: "40대", 6: "50대", 7: "60대 이상",
}
GENDER_LABELS = {0: "남성", 1: "여성"}
POWER_LABELS  = {0: "저소비층", 1: "중소비층", 2: "고소비층"}


# =======================================================================
# [섹션 2] T2 추가 속성 인덱스
# =======================================================================
# 근거: UserModel.py softmax_feature() 코드에서 인덱스 경계 확정
#       속성 레이블: Ali-CCP 스키마 참조 추정값

IDX_OCCUPIED_START,   IDX_OCCUPIED_END   = 62, 64   # VT#7  크기 2
IDX_INTENTIONS_START, IDX_INTENTIONS_END = 16, 27   # VT#2  크기 11
IDX_GEOGRAPHY_START,  IDX_GEOGRAPHY_END  = 27, 38   # VT#3  크기 11
IDX_PROFILE_START,    IDX_PROFILE_END    = 38, 49   # VT#4  크기 11
IDX_GROUP_START,      IDX_GROUP_END      = 49, 60   # VT#5  크기 11
IDX_SHOPS_START,      IDX_SHOPS_END      = 67, 85   # VT#9  크기 18
IDX_BRANDS_START,     IDX_BRANDS_END     = 85, 88   # VT#10 크기 3


# =======================================================================
# [섹션 3] T2 추가 속성 레이블 테이블 (v2 — 의미 있는 자연어로 교체)
# =======================================================================

# ── T2-Step1: is_occupied (VT#7, Ali-CCP user_is_occupied) ──────────────
# 크기 2 → one-hot. 직업 유무 2가지.
IS_OCCUPIED_LABELS = {
    0: "직업 없음(학생/무직)",
    1: "직업 있음(취업자)",
}

# ── T2-Step2: intentions (VT#2, Ali-CCP user_intentions) ────────────────
# 크기 11 → 구매 의도 카테고리
# Ali-CCP + T1 browsing_category 대응 추정
# T1의 카테고리와 같은 상품군이지만 "탐색"이 아닌 "구매 의도"로 표현
INTENTIONS_LABELS = {
    0:  "패션/의류 구매 의향",
    1:  "전자/가전 구매 의향",
    2:  "식품/식료품 구매 의향",
    3:  "뷰티/화장품 구매 의향",
    4:  "스포츠/레저 구매 의향",
    5:  "홈/리빙 구매 의향",
    6:  "도서/문화 구매 의향",
    7:  "유아/육아용품 구매 의향",
    8:  "반려동물 용품 구매 의향",
    9:  "자동차/용품 구매 의향",
    10: "기타 카테고리 구매 의향",
}

# ── T2-Step3: geography (VT#3, Ali-CCP user_geography) ──────────────────
# 크기 11 → 지역/도시 등급
# 중국 도시 티어 구조 참조 (1선 도시: 베이징/상하이/광저우/선전)
GEOGRAPHY_LABELS = {
    0:  "1선 도시 거주(베이징/상하이 등)",
    1:  "신1선 도시 거주(청두/항저우 등)",
    2:  "2선 도시 거주",
    3:  "3선 도시 거주",
    4:  "4선 도시 거주",
    5:  "5선 도시 거주",
    6:  "현급 도시 거주",
    7:  "농촌/향진 거주",
    8:  "화동 지역",
    9:  "화남 지역",
    10: "화북/서부 지역",
}

# ── T2-Step4: profile_type (VT#4, Ali-CCP user_profile) ─────────────────
# 크기 11 → 사용자 프로필 유형
# Ali-CCP에서 직업/라이프스타일 기반 세그먼트로 추정
PROFILE_LABELS = {
    0:  "일반 직장인",
    1:  "전문직(의사/변호사/교수 등)",
    2:  "자영업자/사업가",
    3:  "주부(전업)",
    4:  "대학생",
    5:  "고등학생 이하",
    6:  "은퇴자/시니어",
    7:  "프리랜서/크리에이터",
    8:  "공무원/공기업",
    9:  "IT/테크 종사자",
    10: "기타 직업",
}

# ── T2-Step5: user_group (VT#5, Ali-CCP user_group) ─────────────────────
# 크기 11 → 사용자 충성도/행동 기반 그룹 분류
# 이커머스 고객 세그먼트 표준 분류 참조
USER_GROUP_LABELS = {
    0:  "신규 방문자(첫 방문)",
    1:  "잠재 고객(탐색 단계)",
    2:  "일반 고객(간헐적 구매)",
    3:  "활성 고객(정기 구매)",
    4:  "충성 고객(고빈도 구매)",
    5:  "VIP 고객(최고 등급)",
    6:  "휴면 고객(3개월 이상 비활성)",
    7:  "이탈 위험 고객",
    8:  "재활성 고객(복귀)",
    9:  "시즌성 고객(특정 시기만 구매)",
    10: "기타 그룹",
}

# ── T2-Step6: shops (VT#9, Ali-CCP user_shops, multi-hot 18개) ──────────
# 크기 18 → 사용자가 자주 방문하는 상점 카테고리
# Taobao 상점 유형 분류 참조
SHOPS_LABELS = {
    0:  "패션 전문몰",
    1:  "전자제품 전문몰",
    2:  "공식 브랜드몰",
    3:  "식품/신선식품 전문몰",
    4:  "뷰티/화장품 전문몰",
    5:  "홈/리빙 전문몰",
    6:  "스포츠/아웃도어 전문몰",
    7:  "유아/완구 전문몰",
    8:  "반려동물 전문몰",
    9:  "도서/문화 전문몰",
    10: "자동차 용품 전문몰",
    11: "의약품/헬스 전문몰",
    12: "여행/레저 전문몰",
    13: "농산물/친환경 전문몰",
    14: "명품/럭셔리 전문몰",
    15: "중고/리셀 전문몰",
    16: "해외직구 전문몰",
    17: "종합 멀티몰",
}

# ── T2-Step7: brands (VT#10, Ali-CCP user_brands, 3개) ──────────────────
# 크기 3 → 선호 브랜드 성향
# 중국 이커머스 브랜드 선호도 분류 참조
BRANDS_LABELS = {
    0: "국내(중국) 브랜드 선호",
    1: "해외 글로벌 브랜드 선호",
    2: "브랜드 무관(가성비 우선)",
}


# =======================================================================
# [섹션 4] 각 Step별 속성 추출 함수 (v2 — 구조적 서술 방식)
# =======================================================================

def _extract_t1_base(state_91d: np.ndarray) -> dict:
    """T1 기준 4개 정적 속성 추출."""
    return {
        "category": CATEGORY_LABELS.get(
            int(np.argmax(state_91d[IDX_CATEGORY_START:IDX_CATEGORY_END])), "기타"),
        "age":      AGE_LABELS.get(
            int(np.argmax(state_91d[IDX_AGE_START:IDX_AGE_END])), "연령불명"),
        "gender":   GENDER_LABELS.get(
            int(np.argmax(state_91d[IDX_GENDER_START:IDX_GENDER_END])), "성별불명"),
        "power":    POWER_LABELS.get(
            int(np.argmax(state_91d[IDX_POWER_START:IDX_POWER_END])), "소비층불명"),
    }


def _build_step_sentence(state_91d: np.ndarray, step: int) -> str:
    """
    step에 따라 추가 속성을 개별 문장으로 반환. (v2 수정)

    v1과의 차이:
    - v1: 모든 속성을 쉼표로 이어붙이는 단순 나열
    - v2: 각 속성을 의미 단위 문장으로 분리하여 DistilBERT 이해 향상

    step은 누적 방식: step=3이면 Step1+2+3 속성 모두 포함.
    """
    sentences = []

    if step >= 1:
        # T2-Step1: is_occupied
        idx  = int(np.argmax(state_91d[IDX_OCCUPIED_START:IDX_OCCUPIED_END]))
        label = IS_OCCUPIED_LABELS.get(idx, "직업 정보 불명")
        sentences.append(f"직업 상태는 {label}.")

    if step >= 2:
        # T2-Step2: intentions (구매 의도)
        idx   = int(np.argmax(state_91d[IDX_INTENTIONS_START:IDX_INTENTIONS_END]))
        label = INTENTIONS_LABELS.get(idx, "기타 카테고리 구매 의향")
        sentences.append(f"현재 {label}이 높음.")

    if step >= 3:
        # T2-Step3: geography (지역)
        idx   = int(np.argmax(state_91d[IDX_GEOGRAPHY_START:IDX_GEOGRAPHY_END]))
        label = GEOGRAPHY_LABELS.get(idx, "지역 정보 불명")
        sentences.append(f"{label}.")

    if step >= 4:
        # T2-Step4: profile_type (프로필 유형)
        idx   = int(np.argmax(state_91d[IDX_PROFILE_START:IDX_PROFILE_END]))
        label = PROFILE_LABELS.get(idx, "기타 직업")
        sentences.append(f"직업 프로필은 {label}.")

    if step >= 5:
        # T2-Step5: user_group (사용자 그룹)
        idx   = int(np.argmax(state_91d[IDX_GROUP_START:IDX_GROUP_END]))
        label = USER_GROUP_LABELS.get(idx, "기타 그룹")
        sentences.append(f"고객 등급은 {label}.")

    if step >= 6:
        # T2-Step6: shops (multi-hot — 가장 높은 값의 상점 1개 사용)
        # v2 수정: 임계값 0.1→0.01, top-1만 사용 (항상 값이 나오도록)
        shops_vec = state_91d[IDX_SHOPS_START:IDX_SHOPS_END]
        top_idx   = int(np.argmax(shops_vec))   # 가장 활성화된 상점 1개
        label     = SHOPS_LABELS.get(top_idx, "종합 멀티몰")
        sentences.append(f"주로 {label}을 이용.")

    if step >= 7:
        # T2-Step7: brands (선호 브랜드)
        idx   = int(np.argmax(state_91d[IDX_BRANDS_START:IDX_BRANDS_END]))
        label = BRANDS_LABELS.get(idx, "브랜드 무관")
        sentences.append(f"브랜드 선호 성향은 {label}.")

    return " ".join(sentences)


# =======================================================================
# [섹션 5] T2 핵심 텍스트화 함수 (v2)
# =======================================================================

def textify_state_T2(
    state_91d: np.ndarray,
    group_extractor=None,
    step: int = 1,
) -> str:
    """
    T2 텍스트화 함수 (v2).

    v1과의 핵심 차이:
    - 레이블: "구매의도유형5" → "뷰티/화장품 구매 의향이 높음."
    - 구조: 단순 나열 → 독립 문장 방식 (DistilBERT 이해 향상)
    - shops: 임계값 제거, top-1 항상 반환
    """
    if not isinstance(state_91d, np.ndarray):
        state_91d = np.array(state_91d, dtype=float)
    if state_91d.ndim == 2:
        state_91d = state_91d[0]
    assert 0 <= step <= 7, f"step은 0~7이어야 합니다. 받은 값: {step}"

    base = _extract_t1_base(state_91d)

    # ── T1 기본 프로필 문장 ───────────────────────────────────────────
    profile_sentence = (
        f"고객 프로필: {base['age']} {base['gender']}, "
        f"{base['power']}, "
        f"주요 탐색 카테고리: {base['category']}."
    )

    # ── T2 추가 속성 문장 (step에 따라 누적) ─────────────────────────
    extra_sentences = _build_step_sentence(state_91d, step)

    # ── 동적변수 ──────────────────────────────────────────────────────
    prev_click   = int(state_91d[IDX_PREV_CLICK])
    leave_val    = int(state_91d[IDX_LEAVE_SIGNAL])
    session_step = int(state_91d[IDX_SESSION_STEP])

    leave_label = (
        "이탈 위험 낮음" if leave_val <= 2 else
        "이탈 위험 중간" if leave_val <= 6 else
        "이탈 위험 높음"
    )
    is_cold_start = (session_step == 0)

    if is_cold_start:
        session_sentence = "현재 세션: 첫 접속 (클릭 이력 없음, Cold-start)."
    else:
        if prev_click == 0:
            click_desc = "직전 클릭 없음"
        elif prev_click == 1:
            click_desc = f"직전 {prev_click}번 클릭"
        else:
            click_desc = f"직전 {prev_click}번 클릭(적극 탐색)"
        session_sentence = (
            f"현재 세션: {session_step}번째 추천 진행 중, "
            f"{click_desc}, {leave_label}."
        )

    # ── 집단 트렌드 ───────────────────────────────────────────────────
    group_sentence = ""
    if group_extractor is not None:
        group_text = group_extractor.get_group_text(state_91d)
        if group_text:
            group_sentence = group_text + "."

    # ── 프롬프트 조합 ─────────────────────────────────────────────────
    # 구조: [T1 기본 프로필] [T2 추가 속성] [집단 트렌드] [세션 상태]
    # Cold-start 시: 집단 트렌드를 앞으로 이동하여 강조
    parts = [profile_sentence]

    if extra_sentences:
        parts.append(extra_sentences)

    if is_cold_start and group_sentence:
        parts.insert(1, "[Cold-start: 개인 이력 없음]")
        parts.append(f"유사 고객군 트렌드 참조: {group_sentence}")
    elif group_sentence:
        parts.append(group_sentence)

    parts.append(session_sentence)

    return " ".join(parts)


# =======================================================================
# [섹션 6] prepare_state_T2 — RL 루프 통합 래퍼
# =======================================================================

def prepare_state_T2(
    state_91d: np.ndarray,
    group_extractor=None,
    step: int = 1,
    normalize_dynamic: bool = True,
) -> dict:
    """
    T2 텍스트화 래퍼. t2_run_multiseed_frozen.py의 RL 루프에서 호출.

    Returns:
        {
          "prompt"         : DistilBERT 입력 문자열
          "dynamic_numeric": (3,) numpy array
          "is_cold_start"  : bool
          "step"           : int
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
# [섹션 7] Step 메타데이터
# =======================================================================

T2_STEP_META = {
    0: {"name": "T2-Step0 (=T1 기준선)", "attr": "없음",         "folder": "step0_baseline"},
    1: {"name": "T2-Step1",              "attr": "+is_occupied",  "folder": "step1_occupied"},
    2: {"name": "T2-Step2",              "attr": "+intentions",   "folder": "step2_intentions"},
    3: {"name": "T2-Step3",              "attr": "+geography",    "folder": "step3_geography"},
    4: {"name": "T2-Step4",              "attr": "+profile_type", "folder": "step4_profile"},
    5: {"name": "T2-Step5",              "attr": "+user_group",   "folder": "step5_group"},
    6: {"name": "T2-Step6",              "attr": "+shops",        "folder": "step6_shops"},
    7: {"name": "T2-Step7",              "attr": "+brands",       "folder": "step7_brands"},
}


# =======================================================================
# [섹션 8] 검증 — 프롬프트 품질 확인
# =======================================================================

if __name__ == "__main__":
    print("="*65)
    print("t2_textify.py v2 검증 — 프롬프트 품질 확인")
    print("="*65)

    # 테스트 state
    s = np.zeros(91)
    s[3]  = 1   # category=뷰티/화장품
    s[10] = 1   # age=20대 초반
    s[61] = 1   # gender=여성
    s[66] = 1   # power=고소비층
    s[63] = 1   # is_occupied=직업 있음 (Step1)
    s[19] = 1   # intentions=뷰티/화장품 구매 의향 (Step2)
    s[30] = 1   # geography=2선 도시 (Step3)
    s[42] = 1   # profile=주부 (Step4)
    s[53] = 1   # user_group=활성 고객 (Step5)
    s[71] = 1   # shops=뷰티/화장품 전문몰 (Step6)
    s[86] = 1   # brands=해외 글로벌 브랜드 (Step7)
    s[88] = 3   # prev_click=3
    s[89] = 2   # leave=낮음
    s[90] = 5   # session_step=5

    print()
    for step in range(8):
        meta = T2_STEP_META[step]
        prompt = textify_state_T2(s, group_extractor=None, step=step)
        print(f"[{meta['name']}]")
        print(f"  {prompt}")
        print()

    print("="*65)
    print("v1 vs v2 비교 (Step 2 기준)")
    print("="*65)
    print("""
  v1 프롬프트 (Step2):
    "고객 프로필: 20대 초반 여성, 고소비층, 뷰티/화장품, 직업: 직업 있음, 구매의도: 구매의도유형4."
    → "구매의도유형4" ← DistilBERT 의미 추출 불가

  v2 프롬프트 (Step2):
    "고객 프로필: 20대 초반 여성, 고소비층, 주요 탐색 카테고리: 뷰티/화장품.
     직업 상태는 직업 있음(취업자). 현재 뷰티/화장품 구매 의향이 높음.
     현재 세션: 5번째 추천 진행 중, 직전 3번 클릭(적극 탐색), 이탈 위험 낮음."
    → 모든 단어가 DistilBERT 사전학습 어휘 내 의미 있는 표현 ✅
""")