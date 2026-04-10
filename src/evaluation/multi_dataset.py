"""
Evaluation question set and LLM judge prompt.
"""

EVALUATION_QUESTIONS = [
    {
        "id": "q1",
        "question": "국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.",
        "category": "single_doc",
        "expected_keywords": ["국민연금공단", "이러닝", "요구사항"],
        "expected_orgs": ["국민연금공단"],
        "expected_relevant_docs": 1,
        "reference_answer": "국민연금공단 이러닝 사업의 요구사항을 핵심 항목으로 정리한다.",
        "expected_fields": {
            "발주기관": ["국민연금공단"],
            "요구영역": ["이러닝", "콘텐츠", "관리"],
        },
        "follow_up": {
            "question": "콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘.",
            "category": "follow_up",
            "expected_keywords": ["콘텐츠", "개발", "관리"],
            "expected_relevant_docs": 1,
            "reference_answer": "콘텐츠 개발 관리 요구사항의 세부 항목을 문서 근거로 설명한다.",
        },
    },
    {
        "id": "q2",
        "question": "기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?",
        "category": "single_doc",
        "expected_keywords": ["기초과학연구원", "극저온", "AI", "예측"],
        "expected_orgs": ["기초과학연구원"],
        "expected_relevant_docs": 1,
        "reference_answer": "극저온 시스템 사업에서 AI 기반 예측 요구사항 유무를 명확히 답한다.",
        "expected_fields": {
            "발주기관": ["기초과학연구원"],
            "기술요구": ["AI", "예측", "모니터링"],
        },
        "follow_up": {
            "question": "그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘.",
            "category": "follow_up",
            "expected_keywords": ["모니터링"],
            "expected_relevant_docs": 1,
            "reference_answer": "모니터링 관련 요청사항을 문서 근거로 제시한다.",
        },
    },
    {
        "id": "q3",
        "question": "한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.",
        "category": "single_doc",
        "expected_keywords": ["원자력", "선량", "목적", "추진"],
        "expected_orgs": ["한국원자력연구원", "한국 원자력 연구원"],
        "expected_relevant_docs": 1,
        "reference_answer": "선량 평가 시스템 고도화 사업의 추진 목적을 문서 기반으로 요약한다.",
        "expected_fields": {
            "발주기관": ["한국원자력연구원"],
            "사업목적": ["목적", "고도화", "추진"],
        },
    },
    {
        "id": "q4",
        "question": "고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
        "category": "multi_doc",
        "expected_keywords": ["고려대학교", "광주과학기술원", "비교"],
        "expected_orgs": ["고려대학교", "광주과학기술원"],
        "expected_relevant_docs": 2,
        "reference_answer": "두 기관 사업을 기능, 범위, 요구사항 기준으로 비교한다.",
        "expected_fields": {
            "기관": ["고려대학교", "광주과학기술원"],
            "비교항목": ["기능", "요구사항", "응답 시간"],
        },
        "follow_up": {
            "question": "고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나? 문서를 기반으로 정확하게 답변해 줘.",
            "category": "follow_up",
            "expected_keywords": ["응답 시간", "요구사항"],
            "expected_relevant_docs": 2,
            "reference_answer": "두 기관의 응답 시간 요구사항 유무를 각각 구분해 제시한다.",
        },
    },
    {
        "id": "q5",
        "question": "교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?",
        "category": "cross_doc",
        "expected_keywords": ["교육", "학습"],
        "expected_relevant_docs": 2,
        "reference_answer": "교육/학습 관련 사업을 기관별로 나열한다.",
        "expected_fields": {
            "주제": ["교육", "학습"],
            "출력형식": ["기관", "사업명"],
        },
    },
    {
        "id": "q6",
        "question": "삼성전자가 발주한 반도체 설계 자동화 사업의 요구사항을 알려줘.",
        "category": "out_of_scope",
        "expected_keywords": [],
        "expected_relevant_docs": 0,
        "expected_behavior": "should_decline",
        "reference_answer": "제공 문서에 없는 정보라면 모른다고 답변한다.",
    },
]


LLM_JUDGE_PROMPT = """당신은 RAG 시스템 평가자입니다.
아래 항목을 1~5점으로 평가하고 JSON만 출력하세요.

평가 항목:
1) relevance
2) accuracy
3) faithfulness
4) completeness
5) conciseness

질문: {question}
컨텍스트: {context}
답변: {answer}

JSON 형식:
{{"relevance":1-5,"accuracy":1-5,"faithfulness":1-5,"completeness":1-5,"conciseness":1-5,"reasoning":"..."}}
"""

