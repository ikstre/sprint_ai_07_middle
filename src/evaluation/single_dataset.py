"""
Evaluation question set and LLM judge prompt.
"""

EVALUATION_QUESTIONS = [
{
    "id": "q1",
    "question": "벤처기업협회가 추진하는 2024년 벤처확인종합관리시스템 기능 고도화에서 DAR-002 데이터 이관의 주요 범위와 사용되는 키는 무엇인가요.",
    "category": "single_doc",
    "expected_keywords": [
        "스톡옵션",
        "벤처확인발급번호",
        "사업자등록번호"
    ],
    "expected_orgs": [
        "(사)벤처기업협회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "DAR-002는 스톡옵션 온·오프라인 전체 데이터의 정제·적재와 현 업무단계별 이관, 데이터 오류 정제, 이관 방법과 우선순위 제시, 원천·운영 데이터 저장소 분리, 신규 DB 코드 재정의 등을 포함하며 키는 벤처확인발급번호와 사업자등록번호입니다.",
    "expected_fields": {
        "발주기관": [
            "(사)벤처기업협회"
        ],
        "요구영역": [
            "데이터 이관",
            "키 식별자"
        ]
    },
    "follow_up": {
        "question": "DAR-008 데이터 암호화 요구사항은 개인정보와 연계 데이터에 대해 무엇을 요구하나요.",
        "category": "follow_up",
        "expected_keywords": [
            "암호화",
            "무결성"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "DAR-008은 개인정보 관련 DB데이터의 저장 시 암호화와 암·복호화 권한 관리를 요구하며 연계기관과 주고받는 데이터의 기밀성과 무결성 보장을 요구합니다."
    }
} ,
{
    "id": "q2",
    "question": "(사)부산국제영화제가 발주한 2024년 BIFF & ACFM 온라인서비스 재개발 및 행사지원 사업의 제안서 평가체계와 우선협상대상자 선정 기준을 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "기술능력평가 90점",
        "가격평가 10점",
        "85%"
    ],
    "expected_orgs": [
        "(사)부산국제영화제"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "제안서 평가는 기술능력평가 90점과 가격평가 10점의 총 100점으로 구성되며 기술능력평가 배점한도의 85% 이상을 득점한 자를 협상적격자로 선정하고 합산점수 고득점순으로 우선협상대상자를 결정하며 동점 시 기술점수와 배점이 큰 세부항목 고득점자 순으로 정합니다.",
    "expected_fields": {
        "발주기관": [
            "(사)부산국제영화제"
        ],
        "요구영역": [
            "평가항목",
            "선정기준"
        ]
    },
    "follow_up": {
        "question": "기술능력평가의 정량·정성 부문 배점과 세부 항목 구성을 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "정성평가 80점",
            "정량평가 10점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "기술능력평가는 정량평가 10점(경영상태 4점, 유사실적 4점, 신인도 2점)과 정성평가 80점(계획부문 20점, 기술부문 30점, 시스템공급 10점, 관리부문 10점, 지원부문 10점으로 각 세부 항목 10점씩)으로 구성됩니다."
    }
} ,
{
    "id": "q3",
    "question": "한국대학스포츠협의회 KUSF 체육특기자 경기기록 관리시스템 개발 사업에서 사업추진계획서 제출 기한은 언제인가?",
    "category": "single_doc",
    "expected_keywords": [
        "10일",
        "사업추진계획서"
    ],
    "expected_orgs": [
        "한국대학스포츠협의회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "선정된 사업자는 계약일로부터 10일 이내에 프로젝트 세부추진계획서를 작성하여 제출해야 한다.",
    "expected_fields": {
        "발주기관": [
            "한국대학스포츠협의회"
        ],
        "요구영역": [
            "프로젝트 관리 요구사항",
            "사업추진계획서 작성"
        ]
    },
    "follow_up": {
        "question": "사업추진계획서에는 어떤 단계들이 포함되어야 하는가?",
        "category": "follow_up",
        "expected_keywords": [
            "분석",
            "롤아웃"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "계획에는 분석, 설계, 개발, 테스트, 이행 및 롤아웃 단계가 포함되어야 한다."
    }
} ,
{
    "id": "q4",
    "question": "(재)예술경영지원센터 정보화 용역을 위한 보안관리 주요 대책과 점검 항목을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "자료 인계인수대장",
        "정보보안담당관",
        "완전삭제"
    ],
    "expected_orgs": [
        "(재)예술경영지원센터"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 문서는 비밀유지와 자료반출 금지, 자료 인계인수대장 작성과 자필서명, 정보보안담당관의 전산장비 반입반출 통제와 비밀번호 관리, 인터넷 자료공유 차단과 백신점검, 망분리 운영과 반출 시 저장자료 완전삭제 등의 보안대책과 이행 점검 항목을 제시한다.",
    "expected_fields": {
        "발주기관": [
            "(재)예술경영지원센터"
        ],
        "요구영역": [
            "보안관리",
            "인수인계"
        ]
    },
    "follow_up": {
        "question": "용역사업 완료 시 용역업체가 이행해야 하는 보안 종료 조치는 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "보안확약서",
            "검증필 삭제S/W"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업 완료 시에는 최종 산출물을 대외비로 관리하고 불필요 자료를 삭제·폐기하며 제공 자료 전량을 인계인수대장으로 회수하고 복사본 미보유 보안확약서를 징구하며 전자기록저장매체를 검증필 삭제S/W로 완전삭제 후 승인 반출하고 설정된 계정과 비밀번호를 변경한다."
    }
} ,
{
    "id": "q5",
    "question": "본 입찰의 건명과 입찰보증금 조건을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "종합정보시스템 및 홈페이지",
        "5/100",
        "입찰보증금"
    ],
    "expected_orgs": [
        "2025 구미아시아육상경기선수권대회 조직위원회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "입찰건명은 2025 구미아시아육상경기선수권대회 종합정보시스템 및 홈페이지 등 구축 용역이며 입찰보증금은 낙찰 후 계약 미체결 시 낙찰 금액의 5/100을 현금으로 납부 확약하는 조건입니다.",
    "expected_fields": {
        "발주기관": [
            "2025 구미아시아육상경기선수권대회 조직위원회"
        ],
        "요구영역": [
            "입찰건명",
            "입찰보증금"
        ]
    },
    "follow_up": {
        "question": "이 입찰의 공고 번호와 발표용 PPT 출력물 제출 수량은 어떻게 돼?",
        "category": "follow_up",
        "expected_keywords": [
            "2024 - 6호",
            "10부"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "공고 번호는 조직위 공고 2024 - 6호이며 발표용 PPT 출력물 제출 수량은 10부입니다."
    }
} ,
{
    "id": "q6",
    "question": "BioIN이 발주한 의료기기산업 종합정보시스템 기능개선 사업(2차)의 입찰방식과 입찰 참가자격 요건을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "협상에 의한 계약",
        "공동수급 불가",
        "업종코드 1468"
    ],
    "expected_orgs": [
        "BioIN"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 국가계약법 시행령 제43조 및 제43조의2에 따른 협상에 의한 계약을 적용하며 공동수급은 불가하고, 입찰참가자는 소프트웨어진흥법 제24조의 소프트웨어사업자(컴퓨터관련 서비스 업종코드 1468)로 등록한 중소기업으로서 정보시스템 개발서비스(세부품명번호 8111159901)의 직접생산증명서를 보유해야 한다.",
    "expected_fields": {
        "발주기관": [
            "BioIN"
        ],
        "요구영역": [
            "입찰방식",
            "참가자격"
        ]
    },
    "follow_up": {
        "question": "직접생산증명서의 세부 요구사항은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "8111159901",
            "입찰마감일 전일",
            "유효기간"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "직접생산증명서는 전산업무 소프트웨어 개발 분야의 정보시스템 개발서비스 세부품명번호 8111159901로 발행되어야 하며 입찰마감일 전일까지 발급되어 유효기간 내에 있어야 한다."
    }
} ,
{
    "id": "q7",
    "question": "KOICA가 추진하는 우즈베키스탄 열린 의정활동 사업의 인터넷 의사중계서비스 종합상황판에서 제공되는 상태 정보와 알림 기능을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "종합상황판",
        "회의 공개/비공개",
        "접속자수"
    ],
    "expected_orgs": [
        "KOICA"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "종합상황판은 텍스트·그래픽·소리·아이콘으로 인터넷 중계 현황을 표시하며 회의 공개/비공개 목록과 상태 및 접속자 수, 릴레이 수집/외부송출 신호와 서버 동작상태, 스트리밍 서버 수신·동작상태와 접속자 수, SNS 동작상태와 접속자 수, 회의 진행상태(개의·정회·산회) 및 속개 임박 알림과 시간 변경 알림을 제공하고 기존 수원기관 시스템의 회기정보를 API로 연계합니다.",
    "expected_fields": {
        "발주기관": [
            "KOICA"
        ],
        "요구영역": [
            "기능",
            "인터넷 의사중계서비스"
        ]
    },
    "follow_up": {
        "question": "인터넷 의사중계서비스의 영상 편집 기능에서 지원되는 주요 메타정보 연계와 편집 제어 항목을 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "발언자정보",
            "편집점",
            "전체보기/상세보기"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "영상 편집은 회의기본정보·의사정보·발언자정보·의원정보·대/회/차 정보 연계와 발언유형 관리, 전체보기/상세보기 공개여부 관리, 영상 파일 로딩·볼륨·재생제어·썸네일·오디오 파형, 편집점 등록/수정/삭제/동기화, 메타정보 및 편집정보 불러오기/저장/삭제, 작업로그 생성/적용, 옵션 설정 관리를 지원합니다."
    }
} ,
{
    "id": "q8",
    "question": "경기도 안양시 배드민턴장 및 탁구장 예약시스템 구축용역의 제안서 제출 수량과 업체명 표기 규정은 어떻게 되어 있어?",
    "category": "single_doc",
    "expected_keywords": [
        "10부",
        "원본 1부",
        "표기 금지"
    ],
    "expected_orgs": [
        "경기도 안양시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 제안서는 총 10부를 제출하며 원본 1부에만 업체명과 대표자를 기재하고 인감 날인하며 나머지 9부에는 응모업체를 인지할 수 있는 어떤 표기도 금지되어 있으며 이러한 표기가 발견되면 접수가 무효 처리됩니다.",
    "expected_fields": {
        "발주기관": [
            "경기도 안양시"
        ],
        "요구영역": [
            "제안서 제출수량",
            "식별표시 금지"
        ]
    },
    "follow_up": {
        "question": "수행실적 인정 범위와 제출해야 할 증빙서류는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "최근 3년",
            "문화체육시설 예약시스템",
            "증빙자료"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "수행실적은 공고일 기준 최근 3년 이내 완료한 문화체육시설 예약시스템 또는 정보화사업(시스템, 홈페이지) 구축사업으로서 운영관리 프로그램과 행정감면공동이용시스템 연계 및 반응형 웹사이트 구축 과업을 포함한 실적만 인정되며 단순 하드웨어 납품과 유지보수는 제외되고 실적증명서와 계약서 및 전자세금계산서 등 증빙자료를 제출해야 합니다."
    }
} ,
{
    "id": "q9",
    "question": "평택시 버스정보시스템(BIS) 구축사업의 주된 목적과 노후 장비 및 센터시스템 개선을 통해 달성하려는 운영 측면의 목표는 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "BIS 고도화",
        "대중교통 이용편의",
        "유지보수 효율화"
    ],
    "expected_orgs": [
        "경기도 평택시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 주된 목적은 평택시 시민의 대중교통 이용편의 증진과 서비스 수준 향상을 위한 버스정보시스템(BIS) 고도화이며, 노후 현장장비(BIT) 소프트웨어 개량과 센터시스템 개선을 통해 유지보수 관리의 효율화와 체계적인 시스템 운영을 달성하고자 한다.",
    "expected_fields": {
        "발주기관": [
            "경기도 평택시"
        ],
        "요구영역": [
            "제안개요",
            "배경 및 목표"
        ]
    },
    "follow_up": {
        "question": "사업추진 전략에서 준수해야 할 국가 기준과 미제시 시 따르는 지침은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "국토교통부 기술기준",
            "ITS 업무요령"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업추진 전략에서는 국토교통부가 제정·고시한 대중교통 정보교환 기술기준과 대중교통 기반정보 구축·관리요령을 준수하고 기술기준에서 제시되지 않은 경우 ITS 업무요령을 따른다."
    }
} ,
{
    "id": "q10",
    "question": "경기도사회서비스원 2024년 통합사회정보시스템 운영지원 입찰의 기술능력평가 배점과 산출 방식은 무엇인지 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "기술능력평가",
        "90점",
        "소수점 2자리"
    ],
    "expected_orgs": [
        "경기도사회서비스원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "기술능력평가는 90점 만점으로 소수점 둘째 자리까지 산출하며 평가위원의 최고점과 최저점을 제외한 평균점수로 산출한다.",
    "expected_fields": {
        "발주기관": [
            "경기도사회서비스원"
        ],
        "요구영역": [
            "평가기준",
            "산정방식"
        ]
    },
    "follow_up": {
        "question": "정량적 평가에서 사업수행 실적의 인정 범위와 점수 구간을 요약해줘.",
        "category": "follow_up",
        "expected_keywords": [
            "정보시스템 구축 또는 유지관리",
            "100% 이상 8점",
            "70% 이상 100% 미만 7.2점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업수행 실적은 정보시스템 구축 혹은 유지관리 실적만 인정하며 최근 3년 합산 실적 기준으로 100% 이상 8점, 70% 이상 100% 미만 7.2점, 40% 이상 70% 미만 6.4점, 40% 미만 5.6점으로 평가한다."
    }
} ,
{
    "id": "q11",
    "question": "봉화군 재난통합관리시스템 고도화 사업 제안서의 작성 언어, 분량 제한, 그리고 제안자료 구성 요구사항은 무엇인가요?",
    "category": "single_doc",
    "expected_keywords": [
        "한국어",
        "200매",
        "정량적·정성적·발표자료"
    ],
    "expected_orgs": [
        "봉화군"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "제안서는 반드시 한국어로 200매 이하로 작성하며 제안자료는 정량적, 정성적, 발표자료로 각각 구분해 제출해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "봉화군"
        ],
        "요구영역": [
            "제안서 작성요령",
            "제출구성"
        ]
    },
    "follow_up": {
        "question": "본 사업의 제안요청과 관련된 질의 응답 창구와 제출된 제안서의 소유권 귀속 주체는 누구인가요?",
        "category": "follow_up",
        "expected_keywords": [
            "안전건설과",
            "소유권"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "제안요청 관련 질의는 봉화군 안전건설과가 답변하며 제출된 제안서의 소유권은 봉화군에 있습니다."
    }
} ,
{
    "id": "q12",
    "question": "경희대학교 산학협력단 정보시스템 운영 용역 입찰에서 담합이나 금품 제공이 적발될 경우 입찰참가자격 제한 기간과 계약 관련 조치 사항을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "담합",
        "금품 제공",
        "1년"
    ],
    "expected_orgs": [
        "경희대학교 산학협력단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "경쟁입찰 담합 시 1년 참가 제한과 공정거래위원회 고발이 가능하며 금품·향응 제공으로 유리한 계약 체결 또는 부실 편의 제공 시 2년, 조건 유리화 목적 제공 시 1년, 일반 제공 시 6개월 참가 제한이 적용되고 계약 전 단계별로 낙찰자 결정 취소·계약취소·해제 또는 해지가 이루어질 수 있습니다.",
    "expected_fields": {
        "발주기관": [
            "경희대학교 산학협력단"
        ],
        "요구영역": [
            "입찰제재",
            "계약조치"
        ]
    },
    "follow_up": {
        "question": "실적증명원 제출과 관련한 원본 제출 여부와 제안서 삽입 여부는 어떻게 되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "원본",
            "입찰 등록"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "실적증명원은 원본으로 제출해야 하며 제안서에는 삽입하지 않고 입찰 등록 시 제출합니다."
    }
} ,
{
    "id": "q13",
    "question": "본 사업에서 요구하는 포털 다국어 서비스의 대상과 언어 범위를 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "다국어 서비스",
        "영어",
        "모든 메뉴"
    ],
    "expected_orgs": [
        "고려대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 향후 다국어 환경을 구축하되 영어 서비스를 제공하며 학생과 교수 사용자가 이용하는 포털의 모든 메뉴와 연계 화면을 영어로 제공해야 한다.",
    "expected_fields": {
        "발주기관": [
            "고려대학교"
        ],
        "요구영역": [
            "다국어 서비스",
            "서비스 범위"
        ]
    },
    "follow_up": {
        "question": "국문 기반 다국어 데이터는 어떤 방식으로 관리해야 해?",
        "category": "follow_up",
        "expected_keywords": [
            "데이터베이스",
            "property 파일"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "국문에 해당하는 다국어 데이터는 데이터베이스나 property 파일 등 관리가 용이한 형태로 제공하여야 한다."
    }
} ,
{
    "id": "q14",
    "question": "고양도시관리공사 관산근린공원 다목적구장 홈페이지 및 회원 통합운영 용역의 제안서 보상 여부와 그 근거 조항을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "제안서 보상",
        "제16조",
        "제2023-15호"
    ],
    "expected_orgs": [
        "고양도시관리공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 제안서 보상대상 사업에 해당하지 않아 제안서 보상을 실시하지 않으며 근거는 소프트웨어 사업 계약 및 관리감독에 관한 지침(과학기술정보통신부고시 제2023-15호) 제16조이다.",
    "expected_fields": {
        "발주기관": [
            "고양도시관리공사"
        ],
        "요구영역": [
            "제안서 보상",
            "근거규정"
        ]
    },
    "follow_up": {
        "question": "제안서 관련 붙임 중 보안과 직접 관련된 문서 세 가지를 말해줘.",
        "category": "follow_up",
        "expected_keywords": [
            "보안 서약서",
            "보안특약",
            "비밀유지계약서"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "보안 관련 문서는 용역업체 정보보안 준수사항과 외주 용역사업 보안특약 조항과 보안 서약서가 포함된다."
    }
} ,
{
    "id": "q15",
    "question": "광주과학기술원 RCMS 연계 모듈 변경 사업의 보안사고 위약금 하한 기준과 A급 위반 시 제재를 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "보안 위약금",
        "A급",
        "건당 500만원"
    ],
    "expected_orgs": [
        "광주과학기술원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 보안사고 위약금 하한은 위규 수준에 따라 건당 500만원, 300만원, 100만원으로 적용되며 A급 위반은 국가계약법 시행령 제76조에 따라 부정당업자로 지정되어 입찰참가가 제한됩니다.",
    "expected_fields": {
        "발주기관": [
            "광주과학기술원"
        ],
        "요구영역": [
            "보안 위약금",
            "제재 기준"
        ]
    },
    "follow_up": {
        "question": "보안사고 위약금의 상쇄 가능 여부와 정산 방식은 어떻게 규정되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "상쇄 금지",
            "정산"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "보안 위약금은 다른 요인에 의해 상쇄나 삭감이 불가하며 사업 종료 시 지출금액 조정을 통해 정산됩니다."
    }
} ,
{
    "id": "q16",
    "question": "보안확약서에서 하도급업체의 보안 위반 시 주사업자의 책임과 기관 제재는 어떻게 규정되어 있어?",
    "category": "single_doc",
    "expected_keywords": [
        "하도급",
        "동일한 법적 책임",
        "사업 참여 제한"
    ],
    "expected_orgs": [
        "광주과학기술원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "보안확약서에는 하도급업체에도 동일한 보안사항 준수 책임을 부과하고 위반 시 주사업자가 동일한 법적 책임을 지며 기관의 사업 참여 제한 및 관련 법규에 따른 책임과 손해배상을 감수한다고 규정되어 있습니다.",
    "expected_fields": {
        "발주기관": [
            "광주과학기술원"
        ],
        "요구영역": [
            "보안",
            "하도급"
        ]
    },
    "follow_up": {
        "question": "기밀 보안대책 준수 서약서에서 산출물 전송과 반납에 관한 의무는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "이메일 전송 금지",
            "종료 후 반납"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "기밀 보안대책 준수 서약서에는 보고서 및 산출물을 이메일 등 온라인 전송매체로 전송하지 않고 자료는 관리대장으로 관리하며 사업종료 시 담당자에게 제출 후 PC에서 삭제하고 요청 자료를 반드시 반납해야 한다고 명시되어 있습니다."
    }
} ,
{
    "id": "q17",
    "question": "국가과학기술지식정보서비스 통합정보시스템 고도화 용역의 제안서 평가 항목과 세부 배점 구조를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "평가항목",
        "25점",
        "20점"
    ],
    "expected_orgs": [
        "국가과학기술지식정보서비스"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 용역의 제안평가는 기술 및 기능 25점(기능요구 10점, 보안 5점, 데이터 5점, 운영 3점, 제약 2점), 성능 및 품질 20점(성능 10점, 품질 5점, 인터페이스 5점), 프로젝트 관리 20점(관리방법론 10점, 일정 5점, 개발장비 5점), 프로젝트 지원 10점(품질보증 2점, 시험운영 2점, 교육훈련 2점, 유지관리 1점, 하자보수 1점, 기밀보안 1점, 비상대책 1점)으로 구성된다.",
    "expected_fields": {
        "발주기관": [
            "국가과학기술지식정보서비스"
        ],
        "요구영역": [
            "평가항목",
            "배점"
        ]
    },
    "follow_up": {
        "question": "보안 요구사항의 평가 주안점과 배점은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "보안 요구사항",
            "5점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "보안 요구사항은 관련 표준과 구현 방안을 설계 단계부터 반영해 구체화했는지와 적용 가능성을 평가하며 배점은 5점이다."
    }
} ,
{
    "id": "q18",
    "question": "국가철도공단 철도인프라 디지털트윈 ISP 용역의 정보보안 특약에서 누출금지 대상과 위반 시 제재를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "누출금지",
        "제76조",
        "부정당업체"
    ],
    "expected_orgs": [
        "국가철도공단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "기관 내부 IP주소와 시스템 구성도, 접근권한 정보, 취약점 결과물, 결과물 소스코드, 개인정보 등은 누출금지 대상이며 이를 위반하면 국가계약법 시행령 제76조에 따른 부정당업체 등록과 보안위약금 부과 등 손해배상 책임이 발생합니다.",
    "expected_fields": {
        "발주기관": [
            "국가철도공단"
        ],
        "요구영역": [
            "정보보안",
            "보안위반 제재"
        ]
    },
    "follow_up": {
        "question": "본 용역의 참여인원 보안관리 의무와 교육 주기는 어떻게 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "보안책임관",
            "월1회"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "참여인원 중 보안책임관을 지정해 승인을 받아야 하고 임의 교체가 금지되며 신상변동 시 즉시 보고하고 보안서약서를 제출하며 월1회 정보보안교육을 실시해 발주기관의 확인을 받아야 합니다."
    }
} ,
{
    "id": "q19",
    "question": "국립인천해양박물관 개인정보처리위탁 계약에서 재위탁 제한 조건과 계약 종료 후 개인정보 파기 의무를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "재위탁 제한",
        "개인정보 파기",
        "사전 승낙"
    ],
    "expected_orgs": [
        "국립인천해양박물관"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "을은 갑의 사전 승낙 없이 권리와 의무를 제3자에게 양도하거나 재위탁할 수 없으며 계약 만료 또는 해지 시 보유 개인정보를 관련 고시에 따라 즉시 파기하거나 갑에게 반납하고 파기 결과를 지체 없이 통보해야 한다.",
    "expected_fields": {
        "발주기관": [
            "국립인천해양박물관"
        ],
        "요구영역": [
            "개인정보처리위탁",
            "보안관리"
        ]
    },
    "follow_up": {
        "question": "수탁자 관리감독에서 갑이 점검할 수 있는 개인정보 보호 항목은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "접근 또는 접속 현황",
            "암호화"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "갑은 을에 대해 개인정보 처리 현황, 접근 또는 접속 현황과 대상자, 목적 외 이용제공 및 재위탁 금지 준수 여부, 암호화 등 안전성 확보조치 이행 여부 등 필요한 사항을 점검하고 시정을 요구할 수 있다."
    }
} ,
{
    "id": "q20",
    "question": "국립중앙의료원 차세대 응급의료 상황관리시스템 구축 사업에서 인터넷전화 보안 요구사항을 정리해줘.",
    "category": "single_doc",
    "expected_keywords": [
        "TLS 1.3",
        "RSA 2048",
        "HMAC-SHA256"
    ],
    "expected_orgs": [
        "국립중앙의료원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "인터넷전화는 제어신호에 TLS 1.3 이상과 SHA-256 이상이며 키 교환에 RSA 2048비트를 권고하고 통화내용은 HMAC-SHA1 대신 HMAC-SHA256을 권고하며 모바일 인터넷전화 사용 시 음성데이터만 허용됩니다.",
    "expected_fields": {
        "발주기관": [
            "국립중앙의료원"
        ],
        "요구영역": [
            "보안요구사항",
            "인터넷전화"
        ]
    },
    "follow_up": {
        "question": "품질관리 일반사항에서 시스템 가용성과 운영시간 요구는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "24시간",
            "무중단",
            "가용성"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "시스템은 통상적인 업무시간 동안 가용성을 보장해야 하며 정상상태에서 매일 24시간 무중단으로 운영되어야 합니다."
    }
} ,
{
    "id": "q21",
    "question": "국민연금공단 외주 용역사업 보안특약의 주요 의무와 위반 시 제재 사항을 한 문장으로 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "보안 위약금",
        "국가계약법 시행령 제76조",
        "보안관리계획"
    ],
    "expected_orgs": [
        "국민연금공단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "국민연금공단 외주 용역사업 보안특약에 따라 사업자는 보안관리계획 수립과 누출금지 정보 보호 및 산출물 보안점검·개선을 이행하고 위반 시 위규처리기준에 따른 행정조치와 보안 위약금 부과 및 국가계약법 시행령 제76조에 따른 부정당업체 등록 제재를 받는다.",
    "expected_fields": {
        "발주기관": [
            "국민연금공단"
        ],
        "요구영역": [
            "보안특약",
            "위규처리"
        ]
    },
    "follow_up": {
        "question": "보안위규 처리기준에서 심각 등급 위반 시 요구되는 조치는 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "사업참여 제한",
            "위규자 및 직속 감독자 교체"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "심각 등급 위반 시 사업참여 제한, 위규자 및 직속 감독자 교체, 사유서·경위서 징구, 재발방지 조치계획 제출 및 해당사업 참여자 전체 대상 특별교육 실시가 요구된다."
    }
} ,
{
    "id": "q22",
    "question": "국민연금공단 정보시스템의 서비스 접근 및 전달 분야에서 준수해야 할 웹 표준과 접근성 관련 지침을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "전자정부 웹사이트 품질관리 지침",
        "한국형 웹 콘텐츠 접근성 지침 2.2",
        "HTML 4.01"
    ],
    "expected_orgs": [
        "국민연금공단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "서비스 접근 및 전달 분야에서는 다양한 브라우저와 이용자 접근성을 고려해 표준기술을 준수해야 하며, 관련 세부지침으로 전자정부 웹사이트 품질관리 지침과 한국형 웹 콘텐츠 접근성 지침 2.2 및 모바일 전자정부 서비스 관리 지침을 따르고 웹브라우저 표준은 HTML 4.01/HTML5, CSS 2.1, XHTML 1.0, XML 1.0, XSL 1.0, ECMAScript 3rd와 모바일 웹 콘텐츠 저작 지침 1.0을 적용한다.",
    "expected_fields": {
        "발주기관": [
            "국민연금공단"
        ],
        "요구영역": [
            "서비스 접근 및 전달",
            "웹 표준/접근성"
        ]
    },
    "follow_up": {
        "question": "인터페이스 및 통합 분야에서 타 기관 연계를 위해 적용해야 하는 웹서비스와 데이터 공유 표준은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "SOAP 1.2",
            "WSDL 2.0",
            "RESTful"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "인터페이스 및 통합 분야에서는 SOAP 1.2와 WSDL 2.0 기반의 웹서비스와 RESTful을 적용하고 서비스 발견과 명세는 UDDI v3와 WSDL 2.0을 사용하며 데이터 형식은 XML 1.0을 준수한다."
    }
} ,
{
    "id": "q23",
    "question": "본 사업의 입찰참가 자격과 하도급 기본 제한 사항을 한 문장으로 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "중소 SW사업자",
        "사전승인",
        "50%"
    ],
    "expected_orgs": [
        "국방과학연구소"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 중소 SW사업자만 입찰 가능하며 하도급은 발주기관 사전승인을 받아야 하고 재하도급은 원칙적으로 금지되며 하도급 비율은 사업금액의 50%를 초과할 수 없습니다.",
    "expected_fields": {
        "발주기관": [
            "국방과학연구소"
        ],
        "요구영역": [
            "입찰자격",
            "하도급 제한"
        ]
    },
    "follow_up": {
        "question": "과업심의위원회 개최 여부와 과업변경 시 계약상대자가 해야 할 조치를 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "과업심의위원회",
            "과업변경요청서",
            "계약기간"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업은 과업심의위원회를 개최하며 과업내용 변경과 이에 따른 계약금액·기간 조정이 필요한 경우 계약상대자는 소프트웨어사업 과업변경요청서를 제출해야 합니다."
    }
} ,
{
    "id": "q24",
    "question": "국방과학연구소 대용량 자료전송시스템 고도화 사업의 제안서 제출 형식과 분량·용량 제한, 금지사항을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "PDF",
        "100장",
        "200MB"
    ],
    "expected_orgs": [
        "국방과학연구소"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 제안서는 A4지 규격의 전자문서(PDF)로 작성·제출하며 본문은 양면 기준 100장 이내와 파일 용량 200MB 이내를 준수하고 제안설명 시 홍보용 동영상 활용은 금지됩니다.",
    "expected_fields": {
        "발주기관": [
            "국방과학연구소"
        ],
        "요구영역": [
            "제안서 제출형식",
            "분량 및 용량 제한"
        ]
    },
    "follow_up": {
        "question": "자료 열람을 요청할 때 제안사가 제출해야 하는 서류는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "자료 열람 신청 및 확인서",
            "재직증명서",
            "4대 보험 가입증명서"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "자료 열람 요청 시 제안사는 자료 열람 신청 및 열람 확인서(별지 7호 서식), 재직증명서, 4대 보험 중 하나의 가입증명서(3개월 이내), 주민등록증 또는 운전면허증 신분증을 제출해야 합니다."
    }
} ,
{
    "id": "q25",
    "question": "그랜드코리아레저(주) 그룹웨어의 협업관리 기능에서 제공되는 주요 기능을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "협업관리",
        "업무 상태",
        "Drag&Drop"
    ],
    "expected_orgs": [
        "그랜드코리아레저(주)"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "협업관리는 그룹 생성과 구성원 관리, SNS형 타임라인과 태그·멘션, 댓글과 파일 등록, 즐겨찾기와 폴더 관리, 공지·일정·콘텐츠 모아 보기, 업무 보드와 보드 템플릿 기반의 업무 진행 관리, 요청·처리·검토·종료·보류의 업무 상태 관리, Drag&Drop 상태 변경, 알림 설정, 메일·게시 연계, 뷰와 차트뷰 제공을 지원합니다.",
    "expected_fields": {
        "발주기관": [
            "그랜드코리아레저(주)"
        ],
        "요구영역": [
            "협업관리",
            "업무관리"
        ]
    },
    "follow_up": {
        "question": "협업관리에서 업무 상태 관리와 시각화는 어떻게 제공돼?",
        "category": "follow_up",
        "expected_keywords": [
            "요청·처리·검토·종료·보류",
            "차트뷰"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "업무 상태는 요청·처리·검토·종료·보류 단계로 관리되고 Drag&Drop으로 변경되며 진척과 현황은 전용 뷰와 그래프 형태의 차트뷰로 시각화됩니다."
    }
} ,
{
    "id": "q26",
    "question": "출하승인서에 첨부해야 할 품질확인 문서 패키지의 주요 구성은 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "품질확인 문서",
        "제작공정 및 검사계획서",
        "WPS"
    ],
    "expected_orgs": [
        "기초과학연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "품질확인 문서 패키지는 품질계획서, 제작공정 및 검사계획서, 작업절차서, 용접 관련 WPS·PQR·용접자 자격기록, 자재 인증자료와 검사서류, 불일치사항 종결기록 등으로 구성된다.",
    "expected_fields": {
        "발주기관": [
            "기초과학연구원"
        ],
        "요구영역": [
            "출하승인서",
            "품질문서"
        ]
    },
    "follow_up": {
        "question": "제작공정 및 검사계획서에서 검사점 코드 WP와 HP의 차이는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "WP",
            "HP"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "WP는 검사자 입회가 설정되었지만 합의된 시간에 미입회 시 자체 진행이 가능하고 HP는 검사자 입회 없이는 다음 공정으로 진행할 수 없어 문서상 미입회 확인 전까지 대기해야 한다."
    }
} ,
{
    "id": "q27",
    "question": "나노종합기술원 스마트 팹 서비스 활용체계 구축 관련 설비 온라인 시스템의 품질 및 보안 준수 요구사항의 핵심을 한 문장으로 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "개발보안 가이드",
        "24시간",
        "변경관리"
    ],
    "expected_orgs": [
        "나노종합기술원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 소프트웨어 개발보안 및 보안약점 진단 가이드를 준수하고 품질관리 조직·보증방안을 수립하며 시스템은 통상 업무시간 가용을 보장하고 정상상태에서 24시간 무중단 운영하며 요구사항은 변경관리 절차 승인 후 최종 baseline으로 관리합니다.",
    "expected_fields": {
        "발주기관": [
            "나노종합기술원"
        ],
        "요구영역": [
            "품질 요구사항",
            "보안"
        ]
    },
    "follow_up": {
        "question": "이 사업의 하도급 관리 제약사항 중 대금 지급과 제출 서류 의무를 요약해 줘.",
        "category": "follow_up",
        "expected_keywords": [
            "15일",
            "하도급 대금 지급 비율 명세서"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "하도급은 관련 법률에 따른 사전승인을 전제로 하며 계약상대자는 대가 수령 후 15일 이내 현금 지급과 지급내역 5일 내 통보를 이행하고 하도급 준수실태 보고와 함께 별첨6 서식의 하도급 대금 지급 비율 명세서를 제안서에 포함해 제출해야 합니다."
    }
} ,
{
    "id": "q28",
    "question": "남서울대학교 스마트 정보시스템 활성화 사업에서 개인정보 처리 위탁 계약서에 포함해야 할 필수 조항은 무엇이야.",
    "category": "single_doc",
    "expected_keywords": [
        "개인정보보호법 제26조",
        "재위탁 제한",
        "접근 제한"
    ],
    "expected_orgs": [
        "남서울대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "개인정보 처리 위탁 계약서는 위탁업무 수행 목적 외 처리 금지, 기술적·관리적 보호조치, 위탁업무 목적 및 범위, 재위탁 제한, 개인정보에 대한 접근 제한 등 안전성 확보조치, 관리 현황 점검 등 감독, 수탁자 의무 위반 시 손해배상 등 책임을 포함해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "남서울대학교"
        ],
        "요구영역": [
            "개인정보 위탁",
            "계약조항"
        ]
    },
    "follow_up": {
        "question": "위탁 종료 시 발주자가 확인하고 조치해야 할 사항은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "파기결과 확인",
            "위탁사항 삭제"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "위탁 종료 시에는 사업수행 과정에서 발생한 개인정보의 파기결과를 확인하고 개인정보처리지침의 위탁사항을 삭제해야 합니다."
    }
} ,
{
    "id": "q29",
    "question": "대검찰청 APC-HUB 홈페이지·온라인 교육 시스템의 서비스 접근 및 전달 분야에서 요구되는 웹 표준과 접근성 기준을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "한국형 웹 콘텐츠 접근성 지침 2.1",
        "HTML 4.01/HTML5",
        "표준기술 준수"
    ],
    "expected_orgs": [
        "대검찰청"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 시스템은 다양한 브라우저를 지원하도록 표준기술을 준수하고 서비스 소외계층을 고려하며 전자정부 웹사이트 품질관리 지침과 한국형 웹 콘텐츠 접근성 지침 2.1을 따르고 HTML 4.01/HTML5·CSS 2.1·XHTML 1.0·XML 1.0·ECMAScript 3rd 및 모바일 웹 콘텐츠 저작 지침 1.0을 적용한다.",
    "expected_fields": {
        "발주기관": [
            "대검찰청"
        ],
        "요구영역": [
            "서비스 접근 및 전달",
            "웹 접근성",
            "표준기술"
        ]
    },
    "follow_up": {
        "question": "플랫폼 및 기반구조 분야에서 통신장비가 지원해야 하는 IP 프로토콜 요구사항은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "IPv4",
            "IPv6",
            "동시 지원"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "정보시스템 운영 통신장비는 IPv4와 IPv6를 동시에 지원해야 한다."
    }
} ,
{
    "id": "q30",
    "question": "대전대학교 MILE 사업의 개인정보 처리 위탁에서 수탁자가 준수해야 할 안전성 확보 조치의 주요 범주는 무엇이야.",
    "category": "single_doc",
    "expected_keywords": [
        "관리적 조치",
        "기술적 조치",
        "물리적 조치"
    ],
    "expected_orgs": [
        "대전대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "수탁자는 개인정보보호법 제29조와 시행령 제30조 및 안전성 확보조치 고시에 따라 관리적 조치와 기술적 조치 및 물리적 조치를 시행해야 한다.",
    "expected_fields": {
        "발주기관": [
            "대전대학교"
        ],
        "요구영역": [
            "개인정보보호",
            "보안조치"
        ]
    },
    "follow_up": {
        "question": "위 기술적 조치 중 구체적인 예를 하나 이상 들어줘.",
        "category": "follow_up",
        "expected_keywords": [
            "개인정보 암호화",
            "접속기록"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "예를 들어 수탁자는 접근 권한 관리와 개인정보 암호화 및 접속기록 보관과 위변조 방지와 백신 소프트웨어의 일1회 이상 업데이트를 수행해야 한다."
    }
} ,
{
    "id": "q31",
    "question": "대한상공회의소 기업 재생에너지 지원센터 홈페이지 개편 및 시스템 고도화 사업에서 사회적 약자기업과 정책지원기업에 대한 평가 특례와 제출 서류 요건은 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "사회적 약자기업",
        "만점",
        "증빙서류"
    ],
    "expected_orgs": [
        "대한상공회의소"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "사회적 약자기업과 정책지원기업은 평가등급과 무관하게 만점이 적용되며 이를 증빙할 수 있는 확인서 등 관련 서류를 제출해야 한다.",
    "expected_fields": {
        "발주기관": [
            "대한상공회의소"
        ],
        "요구영역": [
            "평가특례",
            "제출서류"
        ]
    },
    "follow_up": {
        "question": "창업기업의 인정 기준과 기준일은 어떻게 산정돼?",
        "category": "follow_up",
        "expected_keywords": [
            "최근 7년",
            "법인설립등기일"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "창업기업은 입찰공고일 기준 최근 7년 이내 사업 개시 업체로서 법인은 법인등기부상 법인설립등기일을 기준으로 하고 개인사업자는 사업자등록증명서상 사업자등록일을 기준으로 한다."
    }
} ,
{
    "id": "q32",
    "question": "대한장애인체육회가 입찰참가자를 대상으로 수집하는 개인정보의 항목과 수집 목적 및 보유기간과 동의 거부 시 제한 사항을 한 문장으로 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "성명",
        "입찰참가",
        "보유기간"
    ],
    "expected_orgs": [
        "대한장애인체육회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "대한장애인체육회는 성명, 소속, 직위, 주민등록번호, 성별, 연락처, 주소, 학력 및 경력사항 등을 입찰참가에 따른 본인확인 및 대리인 확인을 위해 수집하며 입찰참가에 관한 정보제공의 필요 종결시까지 보유하고 동의를 거부하면 입찰참가가 제한됩니다.",
    "expected_fields": {
        "발주기관": [
            "대한장애인체육회"
        ],
        "요구영역": [
            "개인정보 항목",
            "수집 목적",
            "보유기간"
        ]
    },
    "follow_up": {
        "question": "대한장애인체육회의 고유식별정보 수집의 목적과 항목 및 보유기간과 미동의 시 제한을 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "고유식별정보",
            "정보보안 서약서"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "대한장애인체육회는 입찰참가에 따른 본인 및 대리인 확인 자료로 활용하기 위해 정보보안 서약서 징구 등을 고유식별정보로 수집하며 입찰참가에 관한 정보제공의 필요 종결시까지 보유하고 미동의 시 입찰참가 및 관련 자격이 제한됩니다."
    }
} ,
{
    "id": "q33",
    "question": "보안 위약금 부과 기준에서 A급 위규 발생 시 부과 비중과 추가 제재는 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "A급",
        "10%",
        "부정당업자"
    ],
    "expected_orgs": [
        "대한적십자사 의료원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "A급 위규 발생 시 위약금은 계약금액의 10%가 부과되며 부정당업자 등록이 추가로 적용됩니다.",
    "expected_fields": {
        "발주기관": [
            "대한적십자사 의료원"
        ],
        "요구영역": [
            "보안 위약금",
            "제재기준"
        ]
    },
    "follow_up": {
        "question": "보안 위약금 부과 기준에서 B급 위규의 위약금 비중은 얼마야?",
        "category": "follow_up",
        "expected_keywords": [
            "B급",
            "5%"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "B급 위규의 위약금 비중은 계약금액의 5%입니다."
    }
} ,
{
    "id": "q34",
    "question": "국립민속박물관 민속아카이브 사업의 하도급 관련 주요 제한과 제출 요구사항을 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "하도급",
        "50%",
        "계획서"
    ],
    "expected_orgs": [
        "국립민속박물관"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "국립민속박물관 민속아카이브 사업의 하도급은 입찰금액 대비 합계가 원칙적으로 50%를 초과할 수 없으며 입찰 시 소프트웨어사업 하도급 계획서와 관련 증빙을 제출해야 하고 주관기관 미승인 인력 하도급은 컨소시엄 자사인력으로 대체해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "국립민속박물관"
        ],
        "요구영역": [
            "하도급 제한",
            "제출서류"
        ]
    },
    "follow_up": {
        "question": "과업내용 변경과 과업심의위원회 개최는 어떤 절차와 기준으로 진행돼?",
        "category": "follow_up",
        "expected_keywords": [
            "소프트웨어 진흥법 제50조",
            "과업심의위원회"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "이 사업의 과업내용 변경은 소프트웨어 진흥법 제50조 및 시행령 제47조에 따라 계약상대자가 과업변경요청서를 제출해 과업심의위원회 개최를 요청할 수 있으며 발주기관은 특별한 사정이 없으면 이를 수용해야 합니다."
    }
} ,
{
    "id": "q35",
    "question": "부산관광공사의 경영정보시스템 기능개선 사업에서 협상적격자 선정 기준과 동점 처리 기준은 무엇인가요?",
    "category": "single_doc",
    "expected_keywords": [
        "85%",
        "기술능력 평가",
        "동점"
    ],
    "expected_orgs": [
        "부산관광공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "부산관광공사는 기술능력 평가분야 배점한도의 85% 이상을 협상적격자로 선정하며 합산점수 동점 시 기술능력 평가점수가 높은 자를 우선하고 그마저 동일하면 추첨으로 정합니다.",
    "expected_fields": {
        "발주기관": [
            "부산관광공사"
        ],
        "요구영역": [
            "협상대상자 선정",
            "동점 처리"
        ]
    },
    "follow_up": {
        "question": "이 사업의 평가결과는 어디에 공개되며 어떤 세부내용은 비공개인가요?",
        "category": "follow_up",
        "expected_keywords": [
            "국가종합전자조달시스템",
            "비공개"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "평가결과는 국가종합전자조달시스템에 공개되며 제안서 평가결과의 세부내용과 협상결과는 비공개입니다."
    }
} ,
{
    "id": "q36",
    "question": "사단법인 보험개발원이 요구한 콜센터 챗봇 & Talk 상담솔루션의 핵심 기능과 운영 특성을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "챗봇",
        "24시간 365일",
        "실시간 모니터링"
    ],
    "expected_orgs": [
        "사단법인 보험개발원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 솔루션은 전송대행기관 연동 챗봇으로 24시간 365일 서비스하며 웹기반 플랫폼의 퀵메뉴, 관리자 페이지의 설정·템플릿·메시지·지식사전·말뭉치·인트로 관리, 예약어와 사진송신을 통한 빠른 상담, Talk 상담의 실시간 채팅 모니터링과 이력·통계 관리 기능을 제공합니다.",
    "expected_fields": {
        "발주기관": [
            "사단법인 보험개발원"
        ],
        "요구영역": [
            "콜센터",
            "챗봇"
        ]
    },
    "follow_up": {
        "question": "콜센터 시스템 보안 요구사항에서 계정과 비밀번호 관리 기준은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "1인 1계정",
            "비밀번호 조합규칙"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사용자 식별과 인증은 1인 1계정과 책임 추적성 및 계정관리를 요구하고 비밀번호는 조합규칙 적용과 설정 제한 및 변경 주기 관리가 요구됩니다."
    }
} ,
{
    "id": "q37",
    "question": "중간보고서 제출 주기는 국외 파견기간에 따라 어떻게 규정되어 있어?",
    "category": "single_doc",
    "expected_keywords": [
        "중간보고서",
        "3개월",
        "6개월"
    ],
    "expected_orgs": [
        "사단법인아시아물위원회사무국"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "중간보고서는 국외 파견기간이 3개월 미만이면 제출을 생략하고 3개월 이상 6개월 미만이면 활동 중간에 제출하며 6개월 이상이면 3개월마다 제출합니다.",
    "expected_fields": {
        "발주기관": [
            "사단법인아시아물위원회사무국"
        ],
        "요구영역": [
            "보고서 제출기준",
            "파견기간 구분"
        ]
    },
    "follow_up": {
        "question": "전문가 활동성과 심사표의 평가등급과 평가치는 어떻게 구분되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "평가등급",
            "A(1.00)"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "평가등급은 A(100%), B(90%), C(80%), D(70%), E(60% 미만)이며 평가치는 각각 A(1.00), B(0.90), C(0.80), D(0.70), E(0.60 미만)입니다."
    }
} ,
{
    "id": "q38",
    "question": "서민금융진흥원 서민금융 채팅 상담시스템 구축 사업의 입찰보증금 금액과 보증기간 요건을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "입찰보증금",
        "2.5%",
        "보증기간 30일"
    ],
    "expected_orgs": [
        "서민금융진흥원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "입찰보증금은 입찰금액의 2.5%를 지급보증서 등으로 제안서 제출 시 제출해야 하며 보증기간은 입찰서 제출마감일 다음날부터 30일 이후까지로 정해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "서민금융진흥원"
        ],
        "요구영역": [
            "입찰보증금",
            "보증기간"
        ]
    },
    "follow_up": {
        "question": "낙찰자가 계약을 체결하지 않으면 입찰보증금은 어떻게 처리돼?",
        "category": "follow_up",
        "expected_keywords": [
            "귀속",
            "보증금"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "낙찰자가 계약을 체결하지 않을 경우 해당 보증금액은 서민금융진흥원에 귀속됩니다."
    }
} ,
{
    "id": "q39",
    "question": "서영대학교 산학협력단 차세대 교육 시스템의 전송계층 보안 요구사항 핵심 조치를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "SSL/TLS",
        "암호화",
        "로그인 정보"
    ],
    "expected_orgs": [
        "서영대학교 산학협력단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "사용자 PC부터 웹서버 구간에 SSL, TLS 등 네트워크 구간 암호화를 적용하여 아이디와 패스워드 및 주민등록번호 등 중요 로그인 정보를 스니핑 위협으로부터 보호하고 시스템 간 자료교환의 기밀성, 무결성, 접근제어를 보장해야 한다.",
    "expected_fields": {
        "발주기관": [
            "서영대학교 산학협력단"
        ],
        "요구영역": [
            "보안 요구사항",
            "전송계층 보안"
        ]
    },
    "follow_up": {
        "question": "모바일 응용프로그램 보안 요구사항에서 반드시 준수해야 할 핵심 사항은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "취약점 점검",
            "민감정보 저장 금지"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "모바일 앱은 SQL삽입과 크로스사이트 스크립팅 등 보안약점이 없도록 취약점을 진단하고 제거하며 중요데이터는 저장과 송수신 시 암호화하고 민감한 개인정보와 문서는 단말 내 저장을 금지하며 관련 공공 가이드라인을 준수해야 한다."
    }
} ,
{
    "id": "q40",
    "question": "서울시립대학교가 제시한 평가용 제안서의 작성 및 인쇄 관련 주요 지침을 한 문장으로 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "평가용 제안서",
        "A4",
        "70페이지"
    ],
    "expected_orgs": [
        "서울시립대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "평가용 제안서는 업체명과 로고 등 제안사 표기를 금지하고 A4종 방향을 원칙으로 각 장별 하단 중앙에 일련번호를 부여하며 한글로 작성하고 영문약어표를 제공하며 개념도는 칼라 인쇄가 가능하나 요약서는 70페이지 이내 흑백 인쇄로 작성하고 모호한 표현은 불가능으로 간주되므로 계량화하여 명확히 기술해야 한다.",
    "expected_fields": {
        "발주기관": [
            "서울시립대학교"
        ],
        "요구영역": [
            "제안서 작성지침",
            "인쇄 및 형식"
        ]
    },
    "follow_up": {
        "question": "평가용 제안서 표지 제작기준의 별지서식 번호는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "평가용 제안서",
            "별지서식 16"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "평가용 제안서 표지 제작기준은 별지서식 16번이다."
    }
} ,
{
    "id": "q41",
    "question": "서울특별시 여성가족재단이 공고한 ‘서울 디지털성범죄 안심지원센터’ 사업의 입찰 참가자격 핵심 요건을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "컴퓨터관련서비스사업 1468",
        "8111159901",
        "대기업 참여 불가"
    ],
    "expected_orgs": [
        "서울특별시 여성가족재단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 입찰은 조달청 전자입찰 자격 등록을 완료하고 지방계약법상 자격을 갖춘 소프트웨어사업자(컴퓨터관련서비스사업 1468)이며 공공구매망에 정보시스템개발서비스(8111159901) 또는 소프트웨어유지및지원서비스(8111229901) 직접생산 확인을 받은 중소기업만 참여할 수 있고 대기업 및 상호출자제한기업집단은 참여할 수 없습니다.",
    "expected_fields": {
        "발주기관": [
            "서울특별시 여성가족재단"
        ],
        "요구영역": [
            "입찰자격",
            "제한사항"
        ]
    },
    "follow_up": {
        "question": "공동수급체 구성 제한과 대표사 지정 기준은 어떻게 돼?",
        "category": "follow_up",
        "expected_keywords": [
            "2개 업체 이내",
            "최소 지분율 10%"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "공동수급체는 2개 업체 이내로 구성하고 구성원별 최소 지분율은 10% 이상이어야 하며 참여지분율이 높은 업체를 대표사로 지정합니다."
    }
} ,
{
    "id": "q42",
    "question": "서울특별시 2024년 지도정보 플랫폼 및 전문활용 연계 시스템 고도화 용역의 보안 및 공통 기술표준 준수와 관련한 제출 산출물 요구사항을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "보안서약서",
        "기술적용계획표",
        "정보시스템 구축운영 지침"
    ],
    "expected_orgs": [
        "서울특별시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 행정기관 및 공공기관 정보시스템 구축운영 지침을 포함한 표준을 준수하고 대표자 및 용역참여자 보안서약서, 비밀유지계약서, 업무인수인계 자료관리대장, 최종산출물 보안취약점 점검 및 조치결과서, 기술적용계획표와 기술적용결과표, 누출금지 대상정보 보안관리계획, 자료인수인계 대장 등 제반 보안 및 표준 관련 산출물을 제출해야 한다.",
    "expected_fields": {
        "발주기관": [
            "서울특별시"
        ],
        "요구영역": [
            "보안",
            "표준준수"
        ]
    },
    "follow_up": {
        "question": "웹취약점 점검의 주요 항목과 제출 주기는 어떻게 되며 어떤 증빙을 제출해야 해.",
        "category": "follow_up",
        "expected_keywords": [
            "OWASP 10대",
            "반기별",
            "점검보고서"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "웹취약점 점검은 OWASP 10대 취약점과 국정원 8대 취약점을 대상으로 하며 결과보고서와 조치내역서를 상호협의 후 최소 반기별로 제출해야 한다."
    }
} ,
{
    "id": "q43",
    "question": "서울특별시교육청 지능정보화전략계획(ISP) 수립(2차) 사업에서 보안 위규 발생 시 처분 및 보안 위약금 부과 기준은 무엇인가.",
    "category": "single_doc",
    "expected_keywords": [
        "보안 위약금",
        "계약금액의 10%",
        "부정당업자 등록"
    ],
    "expected_orgs": [
        "서울특별시교육청"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "심각 위규는 부정당업자 등록, 중대 위규는 계약금액의 10%, 보통 위규 2건 이상은 계약금액의 5%, 경미 위규 3건 이상은 계약금액의 3%를 보안 위약금으로 부과하며 다른 요인에 의해 상쇄 또는 삭감되지 않고 별도로 부과됩니다.",
    "expected_fields": {
        "발주기관": [
            "서울특별시교육청"
        ],
        "요구영역": [
            "보안위규 처리",
            "위약금 기준"
        ]
    },
    "follow_up": {
        "question": "누출금지 대상정보에는 어떤 항목들이 포함되나.",
        "category": "follow_up",
        "expected_keywords": [
            "누출금지 대상정보",
            "IP주소"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "누출금지 대상정보에는 정보시스템 내외부 IP주소 현황, 정보통신망 구성도와 시스템 설정 정보, 개별사용자 계정비밀번호 등 접근권한 정보, 취약점 분석평가 결과물과 용역 결과물 소스코드, 법령상 비공개 문서와 개인정보가 포함됩니다."
    }
} ,
{
    "id": "q44",
    "question": "세종테크노파크 인사정보 전산시스템 구축 용역의 하자담보책임기간과 비용 부담 주체 및 하자보수 범위를 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "하자담보책임기간",
        "1년",
        "계약상대자"
    ],
    "expected_orgs": [
        "세종테크노파크"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "하자담보책임기간은 사업종료(검수) 후 1년이며 모든 하자보증 이행 비용은 계약상대자가 부담하고 하자보수 범위는 개발 분야와 도입 장비(HW, SW)를 포함한 전체 시스템입니다.",
    "expected_fields": {
        "발주기관": [
            "세종테크노파크"
        ],
        "요구영역": [
            "하자담보책임",
            "비용부담"
        ]
    },
    "follow_up": {
        "question": "제안서 발표회의 발표자와 발표시간 구성은 어떻게 되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "PM",
            "30분",
            "15분"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "제안서 발표회는 본 사업에 투입될 PM이 직접 발표하며 제안사별 30분 이내로 설명 15분과 질의응답 15분으로 구성됩니다."
    }
} ,
{
    "id": "q45",
    "question": "수협중앙회 강릉어선안전조업국 상황관제시스템 구축 사업에서 공동수급체의 출자비율 변경 사유와 제한을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "출자비율",
        "계약금액 증감",
        "파산·해산·부도"
    ],
    "expected_orgs": [
        "수협중앙회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "출자비율은 발주기관과의 계약내용 변경으로 계약금액이 증감되거나 구성원 중 파산·해산·부도 등으로 당초 이행이 곤란해 연명으로 변경을 요청한 경우에만 변경 가능하며 일부 구성원의 출자비율 전부를 다른 구성원에게 이전할 수 없습니다.",
    "expected_fields": {
        "발주기관": [
            "수협중앙회"
        ],
        "요구영역": [
            "공동수급",
            "계약조건"
        ]
    },
    "follow_up": {
        "question": "공동수급체 구성원이 중도탈퇴하면 잔존 구성원의 계약 이행과 출자비율은 어떻게 처리돼?",
        "category": "follow_up",
        "expected_keywords": [
            "잔존 구성원",
            "출자비율 분할"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "중도탈퇴 시 잔존 구성원이 공동연대하여 계약을 이행하고 필요 요건을 못 갖추면 발주기관 승인을 받아 신규 구성원을 추가할 수 있으며 탈퇴자의 출자비율은 잔존 구성원의 비율에 따라 분할 가산됩니다."
    }
} ,
{
    "id": "q46",
    "question": "수협중앙회의 수산물사이버직매장 시스템 재구축 ISMP 용역에서 정보보안 특수계약조건상 사업 완료 시 산출물과 저장매체 처리 의무를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "대외비",
        "완전 소거",
        "반납"
    ],
    "expected_orgs": [
        "수협중앙회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "용역 종료 시 대외보안이 요구되는 최종 산출물은 대외비 이상으로 관리하고 불필요 자료는 삭제·폐기하며 제공받은 자료와 산출물은 전량 반납하고 복사본을 보관하지 않으며 사업자 보유 저장매체는 완전 소거 후 체크리스트와 자료 미보유 확약서를 제출해야 한다.",
    "expected_fields": {
        "발주기관": [
            "수협중앙회"
        ],
        "요구영역": [
            "정보보안",
            "산출물처리"
        ]
    },
    "follow_up": {
        "question": "이 용역에서 인터넷 접속과 휴대용 저장매체 사용에 대한 제한 사항을 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "인터넷 연결 금지",
            "USB 사용 금지"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업자가 사용하는 PC는 원칙적으로 인터넷 연결이 금지되고 예외적으로 보안통제 하에 허용되며 인가받지 않은 USB 등 휴대용 저장매체 사용은 금지되고 필요 시 정보보안관리자의 승인을 받아야 한다."
    }
} ,
{
    "id": "q47",
    "question": "울산광역시 2024년 버스정보시스템 확대 구축 및 기능개선 용역의 프로젝트 관리 요구사항 중 착수계와 확정설계서 제출 기한 및 주요 제출 항목을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "착수계",
        "7일",
        "60일"
    ],
    "expected_orgs": [
        "울산광역시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "착수계는 계약일로부터 7일 이내에 사업책임자 선임계, 사업수행계획서(전체 예정공정표, 계약일로부터 15일 이내), 사업수행조직도, 보안각서 등으로 제출하며 확정설계서는 착수일로부터 60일 이내에 분야별 구축계획서, 설계내역서, 제작설치도면, 설치위치도 및 시스템·네트워크 전체·세부 구성도 등을 포함해 제출한다.",
    "expected_fields": {
        "발주기관": [
            "울산광역시"
        ],
        "요구영역": [
            "프로젝트관리",
            "제출기한",
            "산출물"
        ]
    },
    "follow_up": {
        "question": "본 사업의 상주 인력 구성과 PM 자격 요건은 어떻게 규정되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "상주",
            "고급기술자"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업관리자는 정보통신공사업법 시행령 제34조에 따른 정보통신 또는 교통 분야 고급기술자 이상 1명을 PM으로 선임해 현장 상주 투입하고 부문별 상주 PL은 센터, 현장, 장비 각 1명씩으로 구성한다."
    }
} ,
{
    "id": "q48",
    "question": "을지대학교 비교과시스템 개발 사업의 무상 하자보수 기간과 하자보수에 포함되는 지원 범위를 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "무상 하자보수",
        "36개월",
        "검사완료일"
    ],
    "expected_orgs": [
        "을지대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "을지대학교 비교과시스템 개발 사업의 무상 하자보수 기간은 검사완료일로부터 36개월이며 지원범위와 지원방법(상주·비상주) 및 지원인원을 포함해 제시하고 문제 발생 시 즉시 조치하며 환경변화에 따른 시스템 변경 최적화와 개발환경 버전 업그레이드 지원 및 장애 처리까지 하자보수 활동에 포함해야 한다.",
    "expected_fields": {
        "발주기관": [
            "을지대학교"
        ],
        "요구영역": [
            "하자보수",
            "운영"
        ]
    },
    "follow_up": {
        "question": "운영 일반 요구사항에서 운영 지원을 위해 반드시 기술해야 하는 핵심 항목은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "운영 일반",
            "보안대책"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "운영 일반 요구사항은 개발 시스템의 원활한 운영을 위해 시스템 정상운영 조건과 조직 및 보안대책을 수립하고 최적화된 프로그램 수행과 관리·운영조직의 역할과 책임 및 제도적 운영관리 대책을 기술하며 데이터베이스 구축 정보의 현행화 방안을 제시하는 것이다."
    }
} ,
{
    "id": "q49",
    "question": "인천공항운영서비스 차세대 ERP 구축 사업에서 검수 및 검사의 실시 기한과 재검수 조건, 확인 범위는 무엇인가요?",
    "category": "single_doc",
    "expected_keywords": [
        "14일",
        "재검수",
        "정상가동"
    ],
    "expected_orgs": [
        "인천공항운영서비스(주)"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "검수는 완료보고서 접수일로부터 14일 이내에 실시하며 요건 불일치 시 지체 없이 보완 후 재검수를 받아야 하고 시스템 납품설치와 정상가동 여부 및 운영에 필요한 관리와 기술지원 등 제반사항을 포함해 확인합니다.",
    "expected_fields": {
        "발주기관": [
            "인천공항운영서비스(주)"
        ],
        "요구영역": [
            "검수 및 검사",
            "프로젝트 관리 요구사항"
        ]
    },
    "follow_up": {
        "question": "검사완료 후 하자보수 기간과 결함 발생 시 조치 시간은 어떻게 정해져 있나?",
        "category": "follow_up",
        "expected_keywords": [
            "12개월",
            "24시간"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "하자보수기간은 검사완료일로부터 12개월이며 결함이나 하자가 발견되면 24시간 이내에 즉시 필요한 조치를 취해 문제를 해결해야 합니다."
    }
} ,
{
    "id": "q50",
    "question": "인천광역시 동구 수도국산달동네박물관 전시해설 시스템 구축 사업의 제안서 평가 방식과 협상적격자 선정 기준을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "기술평가 90%",
        "가격평가 10%",
        "85%"
    ],
    "expected_orgs": [
        "인천광역시 동구"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 제안서 평가는 기술평가 90%와 가격평가 10%를 합산한 종합점수로 하며 기술평가에서는 최고점과 최저점을 제외한 산술평균을 90점 만점으로 환산하고 기술능력평가 배점한도의 85% 이상을 협상적격자로 선정합니다.",
    "expected_fields": {
        "발주기관": [
            "인천광역시 동구"
        ],
        "요구영역": [
            "제안서 평가",
            "배점기준"
        ]
    },
    "follow_up": {
        "question": "정량적 평가 중 사업수행실적은 어떤 기준으로 몇 점을 받을 때 최고 점수를 받는지 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "최근 3년",
            "5건 이상 3.0점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "정량적 평가의 사업수행실적은 공고일 기준 최근 3년간 유사사업 수행건수를 평가하며 5건 이상이면 3.0점을 부여하고 인정범위는 박물관 및 미술관 전시안내 구축 실적만입니다."
    }
} ,
{
    "id": "q51",
    "question": "인천광역시 도시계획위원회 통합관리시스템 구축용역의 계약목적물 지식재산권 귀속 원칙과 상업적 활용 시 필요한 절차를 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "지식재산권",
        "공동소유",
        "협의"
    ],
    "expected_orgs": [
        "인천광역시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 계약목적물 지식재산권은 발주기관과 계약상대자가 원칙적으로 공동소유하며 지분은 별도 정함이 없는 한 균등하고 개발 기여도나 특수성에 따라 협의로 달리 정할 수 있으며 타 용도나 상업적 활용 시 반드시 발주기관과 협의하여야 한다.",
    "expected_fields": {
        "발주기관": [
            "인천광역시"
        ],
        "요구영역": [
            "지식재산권",
            "활용조건"
        ]
    },
    "follow_up": {
        "question": "낙찰자가 소정 기한 내 계약을 체결하지 않으면 입찰보증금과 제재는 어떻게 되니.",
        "category": "follow_up",
        "expected_keywords": [
            "5/100",
            "부정당업자"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "낙찰자가 소정 기한 내 계약을 체결하지 않으면 입찰금액의 5/100에 해당하는 입찰보증금을 인천광역시에 납부하여야 하고 부정당업자로 입찰참가자격 제한처분을 받게 된다."
    }
} ,
{
    "id": "q52",
    "question": "인천광역시가 발주한 인천일자리플랫폼 정보시스템 구축 ISP 수립용역의 보안 위규 수준별 등급과 위약금 부과 기준을 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "A급",
        "총사업비의 5%",
        "부정당업자 등록"
    ],
    "expected_orgs": [
        "인천광역시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 용역의 보안 위규 위약금 기준은 A급(심각 1건) 부정당업자 등록, B급(중대 1건) 총사업비의 5% 부과, C급(보통 2건 이상) 3% 부과, D급(경미 3건 이상) 1% 부과이며 각 위규는 발생 시마다 별도 적용되고 타 항목과 상쇄 없이 사업 종료 시 지출금액 조정을 통해 정산됩니다.",
    "expected_fields": {
        "발주기관": [
            "인천광역시"
        ],
        "요구영역": [
            "보안위반 등급 및 위약금",
            "보안 준수 규정"
        ]
    },
    "follow_up": {
        "question": "사업 착수 시 계약업체와 참여인력이 제출하거나 체결해야 하는 보안 관련 서류와 교육 의무는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "보안서약서",
            "보안교육",
            "비밀유지계약서"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업 착수 시 계약업체와 참여인력은 보안서약서를 제출하고 비밀유지의무 및 제재 내용에 대한 보안교육을 실시하여 결과를 제출하며 대외보안이 필요한 경우 비밀정보 범위와 책임 등을 명시한 비밀유지계약서를 별도로 체결해야 합니다."
    }
} ,
{
    "id": "q53",
    "question": "광주문화재단의 2024년 광주문화예술통합플랫폼 사업 제안서 평가항목과 배점 체계를 한 문장으로 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "총점 100점",
        "정량 10점",
        "정성 80점"
    ],
    "expected_orgs": [
        "재단법인 광주광역시 광주문화재단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 총점은 100점이며 기술능력 평가는 90점으로 정량 10점과 정성 80점으로 구성되고 가격평가는 입찰가격 평점산식에 따른 10점입니다.",
    "expected_fields": {
        "발주기관": [
            "재단법인 광주광역시 광주문화재단"
        ],
        "요구영역": [
            "평가항목",
            "배점"
        ]
    },
    "follow_up": {
        "question": "정량적 평가의 항목별 배점은 어떻게 구성되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "경영상태 3점",
            "수행실적 3점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "정량적 평가는 경영상태 3점, 수행실적 3점, 기술수행능력 3점, 신인도 1점으로 구성됩니다."
    }
} ,
{
    "id": "q54",
    "question": "재단법인 광주연구원의 광주정책연구아카이브(GPA) 시스템 개발 사업의 입찰 참가자격과 참가 제한사항을 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "광주광역시",
        "업종코드 1468",
        "중소기업"
    ],
    "expected_orgs": [
        "재단법인 광주연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 법인등기부상 본점이 광주광역시에 소재하고 소프트웨어사업자(컴퓨터관련서비스사업, 업종코드 1468)로 등록된 중소기업 또는 소상공인이어야 하며 정보시스템개발서비스 직접생산확인증명서와 중소기업소상공인 확인서를 보유하고 대기업·중견기업 및 상호출자제한기업의 참여가 제한되며 공동수급과 하도급이 모두 불가합니다.",
    "expected_fields": {
        "발주기관": [
            "재단법인 광주연구원"
        ],
        "요구영역": [
            "입찰참가자격",
            "참가제한"
        ]
    },
    "follow_up": {
        "question": "협상대상자 선정과 관련한 평가 방식과 합격 기준은 어떻게 되나요?",
        "category": "follow_up",
        "expected_keywords": [
            "기술평가 90점",
            "76.5점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업은 기술평가 90점과 가격평가 10점을 합산해 고득점 순으로 협상하며 기술능력평가가 90점 만점 기준 76.5점 이상인 자를 협상적격자로 선정합니다."
    }
} ,
{
    "id": "q55",
    "question": "본 사업 제안요청서가 요구하는 테스트 계획, 유지관리, 비상대책의 구체적 제시 항목을 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "단위 테스트",
        "하자보수",
        "백업 및 복구"
    ],
    "expected_orgs": [
        "재단법인 한국장애인문화예술원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "제안요청서는 목표시스템에 대해 단위·통합·시스템·인수 테스트의 환경·방법·절차와 구현·테스트 방안 및 개발완료 후 이용·관리운영 방안을 제시하고, 하자보수·유지관리의 계획·조직·절차·범위·기간을 포함하며, 안정적 운영을 위한 백업·복구와 장애대응 등 비상대책을 구체적으로 기술하도록 요구한다.",
    "expected_fields": {
        "발주기관": [
            "재단법인 한국장애인문화예술원"
        ],
        "요구영역": [
            "테스트 계획",
            "유지관리",
            "비상대책"
        ]
    },
    "follow_up": {
        "question": "입찰보증금의 부담 요건은 어떻게 명시되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "2.5%",
            "입찰보증보험증권"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "입찰보증금은 보증금율 2.5% 이상을 입찰보증보험증권으로 납부하도록 명시되어 있다."
    }
} ,
{
    "id": "q56",
    "question": "통합접수시스템 개인정보보호 강화(SER-014)에서 월별 보안점검 항목은 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "SER-014",
        "월별",
        "업무용 PC 보안관리"
    ],
    "expected_orgs": [
        "재단법인경기도일자리재단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "SER-014의 월별 점검 항목은 업무용 PC 보안관리 현황 조사 및 조치보고, 네트워크 장비별 접근 통제 조치보고, 정보시스템별 서비스 포트 현황 조사 및 제거결과 보고, 정보보호시스템 패치적용 및 관련 대장 관리, 정보보호시스템 정책 현행화 및 관련 대장 관리이다.",
    "expected_fields": {
        "발주기관": [
            "재단법인경기도일자리재단"
        ],
        "요구영역": [
            "보안 요구사항",
            "개인정보보호"
        ]
    },
    "follow_up": {
        "question": "해당 요구사항에서 분기별 보안점검 항목은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "분기별",
            "비인가 무선인터넷"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "SER-014의 분기별 점검 항목은 비인가 무선인터넷 시스템 점검 및 조치, 정보시스템 및 네트워크 패스워드 변경 조치와 PC 포함 정보자산 현황 보고, 업무망·DMZ·사용자 대역 내 취약 원격서비스 현황 및 제거 결과 보고, 휴대용 저장매체 무단 반출입 점검, 백신 및 패치관리 서버 점검 및 조치 보고이다."
    }
} ,
{
    "id": "q57",
    "question": "재단법인스포츠윤리센터 LMS 기능개선 사업의 제안서 평가 방식과 협상적격자 선정 기준은 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "기술평가 90%",
        "가격평가 10%",
        "85% 이상"
    ],
    "expected_orgs": [
        "재단법인스포츠윤리센터"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 기술평가 90%와 가격평가 10%의 비중으로 평가하며 기술평가 점수(90점 만점 기준)가 85% 이상인 제안업체에 한해 협상적격자로 선정됩니다.",
    "expected_fields": {
        "발주기관": [
            "재단법인스포츠윤리센터"
        ],
        "요구영역": [
            "평가방식",
            "배점",
            "선정기준"
        ]
    },
    "follow_up": {
        "question": "기술평가의 세부 항목과 배점 구성은 어떻게 되지?",
        "category": "follow_up",
        "expected_keywords": [
            "전략 및 방법론 30점",
            "사업이해도 5점",
            "기능 요구사항 15점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "기술평가는 전략 및 방법론 30점(사업이해도 5점, 추진전략 5점, 적용기술 10점, 개발방법론 10점)과 기술 및 기능 30점(기능 요구사항 15점, 데이터 요구사항 5점, 보안 요구사항 5점)으로 구성됩니다."
    }
} ,
{
    "id": "q58",
    "question": "재단법인충북연구원의 GIS통계 기반 재난안전데이터 분석관리 시스템 사업에서 보안 위약금의 등급별 기준과 위약금 비중을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "A급",
        "부정당업자 등록",
        "계약금액의 5%"
    ],
    "expected_orgs": [
        "재단법인충북연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 보안 위약금은 규정 위반 수준에 따라 A급(심각 1건)은 부정당업자 등록, B급(중대 1건)은 계약금액의 5%, C급(보통 2건 이상)은 계약금액의 3%, D급(경미 3건 이상)은 계약금액의 1%로 부과됩니다.",
    "expected_fields": {
        "발주기관": [
            "재단법인충북연구원"
        ],
        "요구영역": [
            "보안관리",
            "위약금"
        ]
    },
    "follow_up": {
        "question": "심각 위규 발생 시 적용되는 구체적인 처리기준은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "사업참여 제한",
            "중징계",
            "특별 보안교육"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "심각 위규 발생 시에는 사업참여 제한(부정당업체 등록)과 위규자 용역 책임자 중징계 및 재발 방지 조치계획 제출과 위규자 대상 특별 보안교육 실시가 처리기준입니다."
    }
} ,
{
    "id": "q59",
    "question": "전북대학교 JST 공유대학(원) xAPI기반 LRS시스템 구축 사업의 핵심 제약사항과 성과물 소유권 규정을 한 문장으로 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "모듈화",
        "소스 제공",
        "전북지역혁신플랫폼 대학교육혁신본부"
    ],
    "expected_orgs": [
        "전북대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 기능 모듈화와 기 도입 소프트웨어와의 연계·호환성 확보 및 개발된 모든 프로그램의 소스와 이미지 제공을 필수로 하며, 성과품 산출물의 소유권은 준공과 동시에 전북지역혁신플랫폼 대학교육혁신본부에 귀속되고 관련 지식재산권은 공공 목적의 공동 소유이며 침해 시 모든 책임은 사업자에게 있습니다.",
    "expected_fields": {
        "발주기관": [
            "전북대학교"
        ],
        "요구영역": [
            "제약사항",
            "지적재산권"
        ]
    },
    "follow_up": {
        "question": "이 사업에서 따라야 할 데이터 표준화와 공공 기술 표준 준수 기준을 한 문장으로 말해줘.",
        "category": "follow_up",
        "expected_keywords": [
            "행정안전부 고시",
            "LDAP",
            "데이터베이스 표준화"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업은 행정기관 및 공공기관 정보시스템 구축운영지침(행정안전부 고시)과 소프트웨어 개발 보안 가이드를 준수하고 DB 설계 시 행안부 데이터베이스 표준화 지침의 표준용어·코드를 적용하며 연계는 정부디렉토리서비스(LDAP) 연계 표준을 따릅니다."
    }
} ,
{
    "id": "q60",
    "question": "전북특별자치도 정읍시 정읍체육트레이닝센터 통합운영관리시스템 사업에서 계약일로부터 제출해야 하는 핵심 문서와 기한은 무엇이야.",
    "category": "single_doc",
    "expected_keywords": [
        "사업수행계획서",
        "14일",
        "기술적용계획표"
    ],
    "expected_orgs": [
        "전북특별자치도 정읍시"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "사업자는 계약일로부터 14일 이내에 기술적용계획표가 포함된 사업수행계획서와 착수계 등 사업 수행에 필요한 제반 서류를 제출해야 한다.",
    "expected_fields": {
        "발주기관": [
            "전북특별자치도 정읍시"
        ],
        "요구영역": [
            "제출물",
            "기한"
        ]
    },
    "follow_up": {
        "question": "주간보고서와 월간보고서의 제출 주기는 어떻게 지정되어 있어.",
        "category": "follow_up",
        "expected_keywords": [
            "주1회",
            "월1회"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "주간보고서는 주1회 제출하며 월간보고서는 월1회 제출한다."
    }
} ,
{
    "id": "q61",
    "question": "조선대학교 SW중심대학 사업관리시스템 용역에서 누출금지 대상정보의 범주를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "누출금지 대상정보",
        "IP주소",
        "개인정보"
    ],
    "expected_orgs": [
        "조선대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "조선대학교는 정보시스템 내외부 IP현황, 망·시스템 구성도와 설정정보, 계정비밀번호 등 접근권한 정보, 취약점 분석평가 결과와 용역 결과물 및 소스코드, 암호자재와 보안시스템 도입운용 현황, 내부 비공개 문서와 개인정보, 대외비 및 대학이 공개 불가로 판단한 자료를 누출금지 대상정보로 규정한다.",
    "expected_fields": {
        "발주기관": [
            "조선대학교"
        ],
        "요구영역": [
            "보안",
            "비밀관리"
        ]
    },
    "follow_up": {
        "question": "온라인 유지보수를 허용하는 조건과 접속 기록의 보관 기간은 어떻게 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "원격작업 금지 예외",
            "로그 1년"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "원격작업은 원칙적으로 금지이나 서면 동의와 지정 단말기 전용 접속 및 통제시스템 경유 등 보안대책을 준수하는 경우에만 허용되며 접속 로그는 1년 이상 보관한다."
    }
} ,
{
    "id": "q62",
    "question": "중앙선거관리위원회 2025년도 행정정보시스템 위탁운영사업에서 완료보고서 제출의 요건과 형식 및 기한을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "2주전",
        "3부",
        "완료보고서"
    ],
    "expected_orgs": [
        "중앙선거관리위원회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "사업 종료일 2주전까지 검수요청서 및 기술준수 결과표와 함께 개발소스와 산출물·지침서를 정리한 완료보고서를 제출해 승인을 득하고, 최종 승인본은 책자형 3부와 CD 3부로 제작해 사업 종료일까지 제출하며 미승인 시 종료 후 최대 1개월 내 최소인력이 잔류해 최종 산출물을 제출해야 한다.",
    "expected_fields": {
        "발주기관": [
            "중앙선거관리위원회"
        ],
        "요구영역": [
            "완료보고서 제출",
            "제출기한/형식"
        ]
    },
    "follow_up": {
        "question": "형상관리 수행 시 권장되는 도구와 인계 의무는 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "SVN",
            "GIT"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "형상관리는 오픈소스 도구를 권고하며 가급적 SVN 또는 GIT을 사용하고 분기별로 형상관리 대상을 위원회에 인계해 2차 형상관리를 수행하며 사업 종료 시점에 형상관리프로그램과 저장소 데이터를 전체 이관해야 한다."
    }
} ,
{
    "id": "q63",
    "question": "축산물품질평가원 꿀 품질평가 전산시스템 기능개선 사업의 입찰 참가자격과 참여 제한 사항을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "경쟁입찰",
        "8111159901",
        "20억 미만"
    ],
    "expected_orgs": [
        "축산물품질평가원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 경쟁입찰 참가자격과 조달청 입찰참가자격 등록 및 소프트웨어사업자 등록을 갖추고 중소기업·소상공인 확인서와 정보시스템개발서비스(8111159901) 직접생산확인증명서를 보유한 업체만 참여할 수 있으며 소프트웨어 진흥법 제48조 등에 따라 20억 미만 규모로 대기업·중견기업 및 상호출자제한기업집단 소속기업의 참여가 제한됩니다.",
    "expected_fields": {
        "발주기관": [
            "축산물품질평가원"
        ],
        "요구영역": [
            "입찰방식",
            "참가자격"
        ]
    },
    "follow_up": {
        "question": "본 사업에서 대기업과 중견기업의 참여가 제한되는 근거와 금액 기준은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "20억 미만",
            "대기업 참여 제한"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업은 20억 미만 규모로 소프트웨어 진흥법 제48조와 중소 소프트웨어사업자의 사업참여 지원 지침에 따라 대기업과 중견기업 및 상호출자제한기업집단 소속기업의 참여가 제한됩니다."
    }
} ,
{
    "id": "q64",
    "question": "축산물품질평가원 축산물이력관리시스템 개선 사업의 하자보수 지원체계와 범위 및 수행사의 의무를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "하자보수",
        "지원범위",
        "상주"
    ],
    "expected_orgs": [
        "축산물품질평가원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 하자보수는 사업종료 이후 원활한 운영을 위해 상주 또는 비상주 지원인력을 포함한 지원체계를 갖추고 장애 발생 시 즉각적인 원인분석과 복구를 수행하며 개발한 소프트웨어 전 시스템과 보안취약점 조치 및 트러블슈팅 등 정상운영을 위한 기술지원을 포함합니다.",
    "expected_fields": {
        "발주기관": [
            "축산물품질평가원"
        ],
        "요구영역": [
            "하자보수",
            "장애처리"
        ]
    },
    "follow_up": {
        "question": "본 사업의 계약목적물에 대한 지식재산권 귀속 원칙은 어떻게 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "공동 소유",
            "균등 지분"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "계약목적물의 지식재산권은 발주기관과 계약상대자가 공동으로 균등 지분을 원칙으로 소유하되 기여도와 특수성을 고려해 협의로 달리 정할 수 있습니다."
    }
} ,
{
    "id": "q65",
    "question": "과업변경요청서의 처리기간과 변경요청 유형 구분을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "14일",
        "변경요청 유형",
        "과업내용 변경"
    ],
    "expected_orgs": [
        "케빈랩 주식회사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "과업변경요청서의 처리기간은 14일이며 변경요청 유형은 '과업내용 변경'과 '과업내용변경 및 계약금액 조정'으로 구분됩니다.",
    "expected_fields": {
        "발주기관": [
            "케빈랩 주식회사"
        ],
        "요구영역": [
            "과업변경요청서",
            "처리기간"
        ]
    },
    "follow_up": {
        "question": "사업추진체계 수립 시 발주기관 측에서 반드시 명시해야 하는 담당자는 누구야?",
        "category": "follow_up",
        "expected_keywords": [
            "검사 및 감독담당자"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업추진체계에는 발주기관의 검사 및 감독담당자가 반드시 명시되어야 합니다."
    }
} ,
{
    "id": "q66",
    "question": "이 사업의 입찰방식과 적용 규정의 근거를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "협상에 의한 계약",
        "제33382호",
        "제253호"
    ],
    "expected_orgs": [
        "파주도시관광공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "이 사업은 지방자치단체를 당사자로 하는 계약에 관한 법률 시행령(대통령령 제33382호, 2023.4.11.) 제43조와 제44조에 따른 협상에 의한 계약을 적용하며, 행정기관 및 공공기관 정보시스템 구축운영지침(행정안전부 고시 제2023-27호, 2023.4.18.), 지방자치단체 입찰 및 계약 집행기준(행정안전부 예규 제252호, 2023.7.1.), 지방자치단체 입찰시 낙찰자 결정기준(행정안전부 예규 제253호, 2023.7.1.) 및 소프트웨어 진흥법을 준수합니다.",
    "expected_fields": {
        "발주기관": [
            "파주도시관광공사"
        ],
        "요구영역": [
            "입찰방식",
            "적용규정"
        ]
    },
    "follow_up": {
        "question": "소프트웨어 개발사업의 적정 사업기간 산정과 과업내용 확정 절차는 무엇을 따르나.",
        "category": "follow_up",
        "expected_keywords": [
            "제2023-15호",
            "간소화 심의"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업은 과학기술정보통신부 고시 제2023-15호에 따른 적정 사업기간 종합산정서를 첨부하고 지침 제25조 제3항에 따라 정보통신산업진흥원의 과업내용 확정 간소화 심의를 받았습니다."
    }
} ,
{
    "id": "q67",
    "question": "한국가스공사가 발주한 차세대 통합정보시스템 ERP 구축에서 SFR-049 생산공급 시스템 재구축 계획/실적 입력 및 정산 요구사항의 주요 기능을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "데이터 수정",
        "SAP 연계",
        "데이터 이력"
    ],
    "expected_orgs": [
        "한국가스공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "SFR-049는 일일마감 이후 권한과 시간창에 따라 실적 데이터를 수정하고 SAP-생산공급/영업 등 모든 연계 시스템에 자동 반영 상태확인까지 지원하며, 사용자와 DB 레벨에서의 변경 이력 추적과 롤백 및 메신저·메일 기반 진행 알림을 포함해 무인화·개별요금제 등 변화된 프로세스 반영 기능 개선을 요구합니다.",
    "expected_fields": {
        "발주기관": [
            "한국가스공사"
        ],
        "요구영역": [
            "계획/실적 입력 및 정산",
            "데이터 이력 관리"
        ]
    },
    "follow_up": {
        "question": "SFR-051 수율 및 품질 관리 요구사항의 핵심 기능을 요약해줘.",
        "category": "follow_up",
        "expected_keywords": [
            "일 단위 집계",
            "품질 기준"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "SFR-051은 수율 정보를 월 단위에서 일 단위로 집계하고 수율·품질 트렌드 시각화와 조건별 알람을 제공하며 수율 데이터 변경이력 모니터링과 생산공급 시스템과 SAP 간 실적 비교, 미입력·허용범위 초과·표준형식 미준수 등의 품질 기준별 연·월·일 집계 제공을 요구합니다."
    }
} ,
{
    "id": "q68",
    "question": "이 사업의 입찰참가자격과 요구되는 직접생산증명서 세부품명번호는 무엇인가요?",
    "category": "single_doc",
    "expected_keywords": [
        "소프트웨어사업자",
        "1468",
        "8111189901"
    ],
    "expected_orgs": [
        "한국건강가정진흥원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "입찰참가자격은 소프트웨어사업자(컴퓨터관련서비스사업, 업종코드 1468)이며 직접생산증명서는 정보시스템유지관리서비스로 세부품명번호는 8111189901입니다.",
    "expected_fields": {
        "발주기관": [
            "한국건강가정진흥원"
        ],
        "요구영역": [
            "입찰자격",
            "직접생산증명"
        ]
    },
    "follow_up": {
        "question": "입찰 및 제안 관련 문의처 부서와 전화번호는 무엇인가요?",
        "category": "follow_up",
        "expected_keywords": [
            "경영지원부",
            "돌봄지원부",
            "02-3479-7701"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "입찰 관련 문의처는 한국건강가정진흥원 경영지원부 02-3479-7657이고 제안 관련 문의처는 한국건강가정진흥원 돌봄지원부 02-3479-7701입니다."
    }
} ,
{
    "id": "q69",
    "question": "한국교육과정평가원 NCIC 시스템 운영 및 개선 사업의 계약기간 종료일은 언제로 명시되어 있어?",
    "category": "single_doc",
    "expected_keywords": [
        "계약기간",
        "계약체결일",
        "2024.11.15"
    ],
    "expected_orgs": [
        "한국교육과정평가원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "가격 제안서 서식에 따르면 본 사업의 계약기간은 계약체결일부터 2024.11.15.까지입니다.",
    "expected_fields": {
        "발주기관": [
            "한국교육과정평가원"
        ],
        "요구영역": [
            "사업기간",
            "계약조건"
        ]
    },
    "follow_up": {
        "question": "소프트웨어 과업변경요청서의 처리기간은 며칠로 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "처리기간",
            "14일"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "소프트웨어 과업변경요청서의 처리기간은 14일로 명시되어 있습니다."
    }
} ,
{
    "id": "q70",
    "question": "한국농수산식품유통공사 농산물가격안정기금 정부예산회계연계시스템 사업의 협상적격자 선정 기준과 가중치는 무엇인가요.",
    "category": "single_doc",
    "expected_keywords": [
        "기술능력평가 90%",
        "입찰가격평가 10%",
        "85%"
    ],
    "expected_orgs": [
        "한국농수산식품유통공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "기술능력평가 90% 점수가 85% 이상인 자를 대상으로 입찰가격평가 10%를 합산하여 고득점 순으로 협상순위를 결정합니다.",
    "expected_fields": {
        "발주기관": [
            "한국농수산식품유통공사"
        ],
        "요구영역": [
            "평가기준",
            "협상절차"
        ]
    },
    "follow_up": {
        "question": "합산점수가 동일할 때 적용되는 우선순위 규칙은 무엇인가요.",
        "category": "follow_up",
        "expected_keywords": [
            "동점",
            "우선순위"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "합산점수가 동일하면 기술능력 평가점수가 높은 자에게 우선순위를 부여하고 기술능력 평가점수도 동일하면 배점이 큰 항목에서 높은 점수를 얻은 자에게 우선순위를 부여합니다."
    }
} ,
{
    "id": "q71",
    "question": "제안서 제출 파일의 구성, 형식, 제출 방식과 본문 및 요약서 분량 제한을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "PDF",
        "120페이지",
        "30페이지"
    ],
    "expected_orgs": [
        "한국농어촌공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "제안서는 정량제안서·정성제안서·제안요약서를 각각 별도 PDF 파일로 전자조달시스템에 제출하며 본문은 120페이지 이내이고 제안요약서는 30페이지 이내이다.",
    "expected_fields": {
        "발주기관": [
            "한국농어촌공사"
        ],
        "요구영역": [
            "제안서 구성",
            "제출기준"
        ]
    },
    "follow_up": {
        "question": "제안설명회 운영 방식과 추가 자료 제공 제한은 어떻게 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "제안요약서",
            "제안설명회",
            "자료 제공 금지"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "제안설명회는 제안요약서로만 설명하며 제안요약서 외 별도의 제안설명회용 자료나 홍보용 동영상 제공은 금지된다."
    }
} ,
{
    "id": "q72",
    "question": "한국농어촌공사 AFSIS 3단계 협력 입찰의 공동수급 필수 요건을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "3개 이하",
        "10% 이상",
        "대표사"
    ],
    "expected_orgs": [
        "한국농어촌공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 입찰은 공동수급을 허용하며 공동수급업체는 3개 이하로 구성하고 구성원별 계약참여 최소 지분율은 10% 이상이어야 하며 대표사는 다른 구성원보다 참여비율이 높고 PM은 대표사 소속으로 입찰공고일 전부터 재직 중인 자여야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국농어촌공사"
        ],
        "요구영역": [
            "공동수급 요건",
            "제안참가자격"
        ]
    },
    "follow_up": {
        "question": "우선협상대상자 및 낙찰자 결정 기준을 간단히 설명해 줘.",
        "category": "follow_up",
        "expected_keywords": [
            "85%",
            "종합평가점수"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "기술능력평가에서 배점한도의 85% 이상인 자를 협상적격자로 선정하고 기술점수와 가격점수를 합산한 종합평가점수의 고득점 순으로 우선협상대상자를 정하며 동점 시 기술점수와 세부배점이 큰 항목의 점수가 높은 자를 선순위로 결정합니다."
    }
} ,
{
    "id": "q73",
    "question": "청렴계약 및 인권보호 이행서약서에서 금품 제공 금액 구간별 입찰참가자격 제한 기간을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "2억원",
        "3개월",
        "입찰참가자격 제한"
    ],
    "expected_orgs": [
        "한국로봇산업진흥원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "2억원 이상은 2년, 1억원 이상 2억원 미만은 1년, 1천만원 이상 1억원 미만은 6개월, 1천만원 미만은 3개월 동안 입찰에 참가하지 못합니다.",
    "expected_fields": {
        "발주기관": [
            "한국로봇산업진흥원"
        ],
        "요구영역": [
            "청렴서약",
            "제재기간"
        ]
    },
    "follow_up": {
        "question": "경쟁입찰에서 담합이 적발될 경우 유형별 입찰참가 제한 기간을 요약해줘.",
        "category": "follow_up",
        "expected_keywords": [
            "담합",
            "2년"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "담합을 주도하여 낙찰을 받은 경우 2년, 담합을 주도한 경우 1년, 입찰가격을 협정하거나 특정인의 낙찰을 위한 담합의 경우 6월 동안 입찰에 참가할 수 없습니다."
    }
} ,
{
    "id": "q74",
    "question": "한국발명진흥회 사업의 제안서 제출 형식과 분량 권고 및 용어 사용 원칙을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "PDF",
        "500페이지",
        "모호한 표현"
    ],
    "expected_orgs": [
        "한국발명진흥회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "제안서는 PDF 형식으로 제출 가능하며 본문은 500페이지 이내로 작성하고 사용 가능하다 등 모호한 표현은 불가능한 것으로 간주됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국발명진흥회"
        ],
        "요구영역": [
            "제안서 작성지침",
            "표현 원칙"
        ]
    },
    "follow_up": {
        "question": "제안서 효력과 제출 후 변경 가능 여부는 어떻게 규정되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "계약서와 동일한 효력",
            "임의 변경 불가"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "제안서 및 발주자 요구로 수정된 내용은 계약서와 동일한 효력을 가지며 제출된 자료도 제안서와 동일한 효력을 가지고 제출 후 임의 변경은 불가합니다."
    }
} ,
{
    "id": "q75",
    "question": "한국보건산업진흥원이 발주한 본 사업의 입찰방식과 핵심 참가자격 요건을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "협상에 의한 계약",
        "1468",
        "8111159901"
    ],
    "expected_orgs": [
        "한국보건산업진흥원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 국가계약법 시행령 제43조 및 제43조의2에 따른 협상에 의한 계약으로 진행되며, 소프트웨어사업자(업종코드 1468) 등록과 정보시스템 개발서비스 직접생산증명서(세부품명번호 8111159901)를 보유한 중소기업만 입찰에 참여할 수 있습니다.",
    "expected_fields": {
        "발주기관": [
            "한국보건산업진흥원"
        ],
        "요구영역": [
            "입찰방식",
            "참가자격"
        ]
    },
    "follow_up": {
        "question": "공동수급 가능 여부와 중소기업 제한 규정은 어떻게 적용돼?",
        "category": "follow_up",
        "expected_keywords": [
            "공동수급 불가",
            "중소기업"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업은 공동수급이 불가하며 과기정통부 고시 중소 소프트웨어사업자 참여 지원 지침에 따라 입찰 참가자격이 중소기업으로 제한됩니다."
    }
} ,
{
    "id": "q76",
    "question": "한국보육진흥원 연차별 자율 품질관리 시스템 기능개선 사업에서 지식재산권 귀속과 SW 산출물 반출 절차의 핵심 요구사항을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "공동소유",
        "사전승인",
        "누출금지정보"
    ],
    "expected_orgs": [
        "한국보육진흥원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 시스템과 산출물의 지식재산권은 한국보육진흥원(발주부서)과 계약상대자가 공동소유이며, 계약상대자는 누출금지정보를 삭제하고 대표자 확약서를 제출한 뒤에만 활용 가능하고 제3자 제공이나 반출 시 발주부서의 사전승인을 받아야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국보육진흥원"
        ],
        "요구영역": [
            "지식재산권",
            "SW 반출 절차"
        ]
    },
    "follow_up": {
        "question": "하자담보 책임기간은 어떻게 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "1년"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "하자담보 책임기간은 사업의 최종 산출물을 인도한 날을 기준으로 사업 종료일부터 1년입니다."
    }
} ,
{
    "id": "q77",
    "question": "사업수행 일반 요구사항에서 주관사업자는 계약일로부터 며칠 이내에 착수계(사업수행계획서 포함)를 제출해야 하나요?",
    "category": "single_doc",
    "expected_keywords": [
        "착수계",
        "10일"
    ],
    "expected_orgs": [
        "한국사학진흥재단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "주관사업자는 계약일로부터 10일 이내에 착수계(사업수행계획서 포함)를 제출해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국사학진흥재단"
        ],
        "요구영역": [
            "프로젝트관리",
            "일정"
        ]
    },
    "follow_up": {
        "question": "사업 종료 시 제출해야 하는 기능점수 관련 산출물은 무엇인가요?",
        "category": "follow_up",
        "expected_keywords": [
            "기능점수",
            "FP"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "사업 종료 시 응용프로그램의 기능점수(FP)를 종합 집계한 산출내역을 제출해야 합니다."
    }
} ,
{
    "id": "q78",
    "question": "한국사회보장정보원 비밀유지계약에서 비밀유지의무가 면제되는 경우는 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "비밀유지의무의 면제",
        "공지",
        "제3자"
    ],
    "expected_orgs": [
        "한국사회보장정보원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "비밀유지의무는 정보가 제공 이전에 이미 보유된 경우, 당사자의 고의나 과실 없이 공지된 경우, 적법하게 제3자로부터 제공된 경우, 독자적으로 개발된 경우, 제공자가 공개를 허락한 경우, 관련 법규나 정부 요구로 공개되며 사전 서면 통지로 보호조치를 하도록 한 경우에 면제된다.",
    "expected_fields": {
        "발주기관": [
            "한국사회보장정보원"
        ],
        "요구영역": [
            "비밀유지",
            "면책사유"
        ]
    },
    "follow_up": {
        "question": "비밀정보가 포함된 자료의 반환 및 폐기 의무는 어떻게 정해져 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "반환",
            "폐기"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "상대방이 요청하면 비밀정보가 포함된 모든 자료와 유체물을 즉시 반환하거나 상대방의 선택에 따라 폐기하고 그 폐기 증명서를 제공해야 한다."
    }
} ,
{
    "id": "q79",
    "question": "한국산업단지공단의 산단 안전정보시스템 1차 구축 용역의 입찰참가 자격과 중소기업 제한 조건을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "20억원 미만",
        "중소 소프트웨어 사업자",
        "소프트웨어 진흥법 제48조"
    ],
    "expected_orgs": [
        "한국산업단지공단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 총 사업금액이 20억원 미만으로 소프트웨어 진흥법 제48조 등에 따라 대기업 및 중견기업의 참여가 제한되며 중소 소프트웨어 사업자만 입찰참가가 가능하고 나라장터 해당 업종(컴퓨터관련서비스 1468, 정보통신공사업 0036) 등록과 중소기업자 확인 및 정보시스템개발서비스 직접생산확인증명서 요건을 충족해야 한다.",
    "expected_fields": {
        "발주기관": [
            "한국산업단지공단"
        ],
        "요구영역": [
            "입찰자격",
            "참여제한"
        ]
    },
    "follow_up": {
        "question": "이 사업의 하도급 사전승인 요건과 하도급 비율 제한은 어떻게 규정되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "사전 승인",
            "50%"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업의 하도급은 소프트웨어 진흥법 제20조의3 및 관련 지침에 따라 계약 전에 발주기관의 사전 승인을 받아야 하며 하도급 비율은 사업금액의 50%를 초과할 수 없고 재하도급은 원칙적으로 불허된다."
    }
} ,
{
    "id": "q80",
    "question": "한국산업인력공단 RFID기반 국가자격 시험 결과물 스마트 관리시스템 사업의 정량적 평가 중 경영상태 평가 기준과 등급별 배점을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "신용평가등급",
        "배점의 100%"
    ],
    "expected_orgs": [
        "한국산업인력공단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 정량적 평가에서 경영상태는 신용평가등급에 따라 AAA~BBB0는 배점의 100%, BBB-~BB-는 배점의 95%, B+~B-는 배점의 90%, CCC+ 이하 등은 배점의 70%로 산정됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국산업인력공단"
        ],
        "요구영역": [
            "평가기준",
            "정량평가"
        ]
    },
    "follow_up": {
        "question": "국가종합전자조달시스템에서 신용평가등급 확인서가 확인되지 않은 경우에는 어떻게 평가해?",
        "category": "follow_up",
        "expected_keywords": [
            "최저등급"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "신용평가등급 확인서가 확인되지 않은 경우에는 최저등급으로 평가하며 유효기간 시작일 또는 만료일이 입찰공고일인 경우에도 유효한 것으로 평가합니다."
    }
} ,
{
    "id": "q81",
    "question": "본 사업의 하자보증기간과 최종 검사 요청 시 제출해야 할 서류는 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "하자보증기간",
        "12개월",
        "결과보고서"
    ],
    "expected_orgs": [
        "한국생산기술연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 과업의 하자보증기간은 용역완료일로부터 12개월이며 최종 검사 요청 시 최종 성과물과 과업 진행과정 및 결과내용을 수록한 결과보고서를 함께 제출해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국생산기술연구원"
        ],
        "요구영역": [
            "검사검수",
            "하자보증"
        ]
    },
    "follow_up": {
        "question": "본 사업의 입찰 참여가 제한되는 대상은 누구야?",
        "category": "follow_up",
        "expected_keywords": [
            "상호출자제한기업집단",
            "대기업",
            "중견기업"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "본 사업은 소프트웨어 진흥법 제48조 및 관련 지침에 따라 상호출자제한기업집단 소속회사와 20억원 미만 사업 기준에 따라 대기업 및 중견기업인 소프트웨어 사업자의 입찰 참여가 제한됩니다."
    }
} ,
{
    "id": "q82",
    "question": "EIP3.0 고압가스 안전관리 시스템 구축에서 요구하는 프레임워크 기반 개발의 핵심 요건을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "EIP3.0",
        "프레임워크",
        "공통모듈"
    ],
    "expected_orgs": [
        "한국생산기술연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "모든 시스템은 EIP3.0에서 활용 중인 프레임워크로 개발하며 공통모듈·권한관리 등 응용SW 기본구조를 동일하게 적용하고 사업 착수 시 성능 저하 이슈를 점검해 필요 시 프레임워크를 최적화하며 설계 이전에 공통 기능을 도출·파일럿으로 선도 개발하고 전자정부 프레임워크 기반 응용 아키텍트를 투입해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국생산기술연구원"
        ],
        "요구영역": [
            "프레임워크",
            "개발기준"
        ]
    },
    "follow_up": {
        "question": "프레임워크 수정 관련 비용과 변경 절차는 어떻게 규정되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "사업자 부담",
            "협의"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "프레임워크 수정 비용은 사업자가 부담하며 추가·변경 요건 구현은 전 공정에 걸쳐 주관기관과 협의해 진행하고 개발 관리 절차에 따라 배포해야 합니다."
    }
} ,
{
    "id": "q83",
    "question": "한국수자원공사의 건설통합시스템(CMS) 고도화 사업의 기간과 적정 개발기간 산정 근거를 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "12개월",
        "2024년 6월 ~ 2025년 5월",
        "1,380.7 FP"
    ],
    "expected_orgs": [
        "한국수자원공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 추진 기간은 2024년 6월부터 2025년 5월까지이며 기능점수 1,380.7 FP와 1인 생산성 22 FP/MM 및 투입 인력 5.5명을 근거로 산정된 적정 개발기간은 12개월입니다.",
    "expected_fields": {
        "발주기관": [
            "한국수자원공사"
        ],
        "요구영역": [
            "사업기간",
            "개발기간 산정"
        ]
    },
    "follow_up": {
        "question": "이 사업의 예상 사용자 규모는 내부 직원과 일반 국민 각각 몇 명이야.",
        "category": "follow_up",
        "expected_keywords": [
            "3,000명",
            "9,000명"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "예상 사용자수는 내부 직원 3,000명과 일반 국민 또는 기업 9,000명입니다."
    }
} ,
{
    "id": "q84",
    "question": "한국수자원공사 수도사업장 통합 사고분석솔루션 시범구축 용역의 보안정책 위반 시 위약금 부과 기준과 금액을 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "보안 위약금",
        "A급",
        "3천만원"
    ],
    "expected_orgs": [
        "한국수자원공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "K-water의 보안정책 위반 시 붙임 2의 처리기준에 따라 연차사업 단위로 위약금이 부과되며 A급은 부정당업자등록, B급은 3천만원, C급은 2천만원, D급은 1천만원이 적용됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국수자원공사"
        ],
        "요구영역": [
            "보안정책",
            "위약금"
        ]
    },
    "follow_up": {
        "question": "보안 위약금 부과와 관련된 위규 수준의 정의와 조건을 알려줘.",
        "category": "follow_up",
        "expected_keywords": [
            "심각 1건",
            "경미 3건 이상"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "위규 수준은 A급이 심각 1건, B급이 중대 1건, C급이 보통 2건 이상, D급이 경미 3건 이상으로 정의됩니다."
    }
} ,
{
    "id": "q85",
    "question": "한국수자원공사가 추진하는 용인 첨단 시스템반도체 국가산단 용수공급사업의 타당성 평가에서 반드시 검토해야 할 핵심 항목과 준거 지침을 한 문장으로 정리해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "건설기술진흥법 시행령 제81조",
        "국토교통부고시 제2016-291호",
        "경제성 분석"
    ],
    "expected_orgs": [
        "한국수자원공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 타당성 평가는 건설기술진흥법 시행령 제81조와 K-water 건설기술관리규정 제17조 및 국토교통부고시 제2016-291호에 따라 개략사업비 비교, 운영·유지관리비, 단계별 수요·공급에 따른 경제성(B/C·NPV·IRR 및 민감도)과 사회·환경·정책성, 전략환경영향평가 반영, 민원 및 장애요인 파악 등을 종합 검토해야 한다.",
    "expected_fields": {
        "발주기관": [
            "한국수자원공사"
        ],
        "요구영역": [
            "타당성 평가",
            "기본계획"
        ]
    },
    "follow_up": {
        "question": "대형공사 집행기본계획서의 제출 시한은 과업착수 후 언제로 규정돼 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "대형공사",
            "2개월내에"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "대형공사 집행기본계획서는 동법 시행규칙 제78조 제1항에 따라 과업착수 후 2개월내에 감독원에게 제출해야 한다."
    }
} ,
{
    "id": "q86",
    "question": "본 용역의 성과품 제출 기한과 제출해야 할 핵심 항목은 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "14일 이내",
        "소스코드",
        "시스템 구성도"
    ],
    "expected_orgs": [
        "한국수자원조사기술원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "계약 종료일로부터 14일 이내에 성과품·산출물·프로그램과 소스를 수록한 매체와 소스코드 및 시스템 구성도를 제출해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국수자원조사기술원"
        ],
        "요구영역": [
            "제출기한",
            "산출물"
        ]
    },
    "follow_up": {
        "question": "제안서의 작성 형식과 권고 규격은 무엇으로 지정되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "A4",
            "pdf"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "제안서는 A4지 규격의 전자문서(pdf)로 작성하는 것을 권고하며 A4 종 방향을 원칙으로 하되 부득이한 경우 A4 횡 또는 기타 용지를 일부 사용할 수 있습니다."
    }
} ,
{
    "id": "q87",
    "question": "한국수출입은행이 발주한 모잠비크 마푸토 ITS 구축사업의 과업수행 원칙과 현지 상주 요건을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "PM",
        "4M/M(120일)"
    ],
    "expected_orgs": [
        "한국수출입은행"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 과업은 발주자 지시에 따라 PM이 모든 회의와 연락을 주도하는 현장중심 수행을 원칙으로 하며 수원국 기관과 합의를 전제로 제안서 지정 1인 이상이 4M/M(120 calendar-day) 이상 현지 상주하고 주요 인력은 최소 현지출장횟수를 준수해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한국수출입은행"
        ],
        "요구영역": [
            "과업수행 원칙",
            "현지상주 요건"
        ]
    },
    "follow_up": {
        "question": "기후 분야 분책의 현지출장 최소 요건은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "2회"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "기후 분야 분책은 최소 2회 이상 현지출장이 필요합니다."
    }
} ,
{
    "id": "q88",
    "question": "한국어촌어항공단 ERPGW 기능 고도화 사업의 제안서 평가에서 기술·가격 비중과 종합평가 동점자 처리 기준은 무엇인가?",
    "category": "single_doc",
    "expected_keywords": [
        "기술평가 90%",
        "가격평가 10%",
        "동점자 처리"
    ],
    "expected_orgs": [
        "한국어촌어항공단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업의 제안서 평가는 기술평가 90%와 가격평가 10%의 비중으로 종합평가하며 동점 시 기술평가점수가 높은 업체를 우선하고 그것도 동일하면 배점이 높은 평가항목에서 점수가 높은 업체를 선정한다.",
    "expected_fields": {
        "발주기관": [
            "한국어촌어항공단"
        ],
        "요구영역": [
            "평가방법",
            "동점자 처리"
        ]
    },
    "follow_up": {
        "question": "기술평가위원회 구성 및 기술평가 점수 산출 방식은 어떻게 되나?",
        "category": "follow_up",
        "expected_keywords": [
            "8인 이하",
            "최고점과 최저점 제외"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "기술평가는 주관기관이 8인 이하 전문가로 기술평가위원회를 구성해 각 위원의 점수를 산술평균하여 90점 만점으로 합산하며 위원이 7인 이상이면 최고점과 최저점 각 1개를 제외해 평균을 산출한다."
    }
} ,
{
    "id": "q89",
    "question": "한국연구재단의 2024년 기초학문자료센터 사업에서 보안 위규자 처리기준의 구분과 각 구분별 주요 조치 내용을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "보안 위규자 처리기준",
        "심각",
        "중징계"
    ],
    "expected_orgs": [
        "한국연구재단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "보안 위규자 처리기준은 심각·중대·보통·경미로 구분되며 심각은 사업 참여 제한과 위규자 및 직속 감독자 중징계와 재발 방지 조치계획 제출 및 특별보안교육 실시가 적용되고 중대는 위규자 및 직속 감독자 중징계와 재발 방지 조치계획 제출 및 특별보안교육 실시가 적용되며 보통은 경징계와 사유서·경위서 징구 및 특별보안교육 실시가 적용되고 경미는 경미한 관리소홀 등에 대한 경미 조치가 적용됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국연구재단"
        ],
        "요구영역": [
            "보안관리",
            "처리기준"
        ]
    },
    "follow_up": {
        "question": "심각 구분의 보안 위규 사항에는 어떤 행위들이 포함돼?",
        "category": "follow_up",
        "expected_keywords": [
            "비밀 정보 유출",
            "해킹"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "심각 구분에는 비밀 및 대외비급 정보 유출이나 유출시도, 관련 시스템에 대한 해킹 및 해킹시도, 시스템 구축 결과물 외부 유출, 시스템 내 인위적인 악성코드 유포가 포함됩니다."
    }
} ,
{
    "id": "q90",
    "question": "한국연구재단 UICC 기능개선 사업의 입찰 참가자격과 공동수급·하도급 주요 제한을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "중소기업",
        "상호출자제한기업집단",
        "50%"
    ],
    "expected_orgs": [
        "한국연구재단"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 사업은 중소기업 및 소상공인만 입찰 가능하고 상호출자제한기업집단과 대기업·중견기업은 참여가 제한되며 공동수급은 5개 이하 구성원별 10% 이상 지분의 공동이행만 허용되고 하도급은 발주기관 사전승인 하에 소프트웨어사업금액의 50%를 초과할 수 없습니다.",
    "expected_fields": {
        "발주기관": [
            "한국연구재단"
        ],
        "요구영역": [
            "입찰자격",
            "공동수급",
            "하도급"
        ]
    },
    "follow_up": {
        "question": "본 사업의 기술·가격 평가 비율과 협상적격자 선정 기준은 무엇이야.",
        "category": "follow_up",
        "expected_keywords": [
            "기술평가 90%",
            "85%"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "평가비율은 기술평가 90%와 가격평가 10%이며 기술능력평가 배점한도의 85% 이상 득점자를 협상적격자로 선정합니다."
    }
} ,
{
    "id": "q91",
    "question": "한국원자력연구원 선량평가시스템 고도화 사업에서 보안위반 등급별 제재와 위약금 하한은 어떻게 정해져 있어?",
    "category": "single_doc",
    "expected_keywords": [
        "A급",
        "500만원",
        "부정당업자"
    ],
    "expected_orgs": [
        "한국원자력연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "보안위반은 A급·B급·C급·D급으로 구분되며 위약금 하한은 건당 500만원·300만원·100만원으로 정해지고 A급은 국가계약법 시행령 제76조에 따라 부정당업자로 지정되어 입찰참가가 제한됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국원자력연구원"
        ],
        "요구영역": [
            "보안위반 등급",
            "위약금",
            "제재"
        ]
    },
    "follow_up": {
        "question": "보안 위약금의 부과 및 정산 원칙은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "상쇄 금지"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "보안 위약금은 다른 요인에 의한 상쇄나 삭감 없이 별도로 부과되며 사업 종료 시 지출금액 조정을 통해 정산됩니다."
    }
} ,
{
    "id": "q92",
    "question": "한국재정정보원 e나라도움 업무시스템 웹 접근성 컨설팅 입찰에서 사업 실적 증명서 제출과 인정 기준은 무엇이야?",
    "category": "single_doc",
    "expected_keywords": [
        "실적증명서",
        "페이지 표시"
    ],
    "expected_orgs": [
        "한국재정정보원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 입찰에서는 실적증명서와 계약서 등 증거서류로 확인이 불가능한 실적은 인정하지 않으며 실적증명자료는 붙임으로 첨부하고 실적증명첨부서류의 페이지를 명시하여 주요사업실적 비고란에 해당 페이지를 표시해야 한다.",
    "expected_fields": {
        "발주기관": [
            "한국재정정보원"
        ],
        "요구영역": [
            "사업실적 증명",
            "제출요건"
        ]
    },
    "follow_up": {
        "question": "공동수급체의 출자비율은 어떤 경우에 변경할 수 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "출자비율",
            "변경 사유"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "공동수급체의 출자비율은 발주기관과의 계약내용 변경으로 계약금액이 증감되거나 구성원의 파산·해산·부도·법정관리·워크아웃·중도탈퇴 등의 사유로 당초 이행이 곤란하여 연명으로 변경을 요청한 경우에만 변경할 수 있으며 일부 구성원의 출자비율 전부를 다른 구성원에게 이전할 수는 없다."
    }
} ,
{
    "id": "q93",
    "question": "한국전기안전공사 관제시스템 보안 모듈 개발 용역의 보안 위약금 부과 기준에서 등급별 위규 수준과 위약금 비중을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "A급",
        "계약금액의 5%",
        "부정당업자 등록"
    ],
    "expected_orgs": [
        "한국전기안전공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "보안 위약금 부과 기준은 A급은 심각 1건으로 부정당업자 등록, B급은 중대 1건으로 계약금액의 5%, C급은 보통 2건 이상으로 계약금액의 3%, D급은 경미 3건 이상으로 계약금액의 1%가 부과됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국전기안전공사"
        ],
        "요구영역": [
            "보안 위약금",
            "위규 등급"
        ]
    },
    "follow_up": {
        "question": "이 용역에서 누출금지 대상 정보에는 어떤 항목들이 포함돼?",
        "category": "follow_up",
        "expected_keywords": [
            "IP주소",
            "계정비밀번호"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "누출금지 대상 정보에는 내외부 IP주소 현황, 세부 시스템 구성 및 망 구성도, 사용자 계정 비밀번호 등 접근권한 정보, 취약점 분석평가 결과물, 용역사업 결과물, 보안시스템 및 네트워크 장비 설정 정보, 비공개 대상 내부문서, 개인정보, 비밀 및 대외비, 그리고 사장이 공개 불가로 판단한 자료가 포함됩니다."
    }
} ,
{
    "id": "q94",
    "question": "IP-NAVI 해외지식재산센터 사업관리 시스템 기능개선 사업의 과제명과 발주기관 및 제안서 주요 목차를 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "IP-NAVI",
        "기능개선",
        "제안서 목차"
    ],
    "expected_orgs": [
        "한국지식재산보호원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "과제명은 'IP-NAVI 해외지식재산센터 사업관리 시스템 기능개선'이며 발주기관은 한국지식재산보호원이고 제안서 주요 목차는 일반현황, 전략 및 방법론, 기술 및 기능, 성능 및 품질, 프로젝트 관리, 프로젝트 지원, 기타사항이다.",
    "expected_fields": {
        "발주기관": [
            "한국지식재산보호원"
        ],
        "요구영역": [
            "과제명",
            "제안서 목차"
        ]
    },
    "follow_up": {
        "question": "제안서 세부 작성지침에서 보안 요구사항에 포함해야 할 내용은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "보안 요구사항",
            "보안기술"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "보안 요구사항은 시스템과의 관련성 분석과 적용할 보안기술·표준·제안방안을 구체적으로 제시해야 한다."
    }
} ,
{
    "id": "q95",
    "question": "한국철도공사 운행정보기록 자동분석시스템 사업에서 사업수행계획서 제출 기한과 검수 기간은 각각 어떻게 정해져 있어?",
    "category": "single_doc",
    "expected_keywords": [
        "사업수행계획서",
        "10일",
        "14일"
    ],
    "expected_orgs": [
        "한국철도공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "사업수행계획서는 계약일로부터 10일 이내 제출해야 하며 검수는 완료보고서 접수일로부터 14일 이내에 실시됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국철도공사"
        ],
        "요구영역": [
            "제출기한",
            "검수기간"
        ]
    },
    "follow_up": {
        "question": "교육환경 관련 소요비용의 부담 주체는 누구로 명시되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "제안사",
            "비용부담"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "교육환경과 관련된 소요비용은 제안사에서 부담함을 원칙으로 합니다."
    }
} ,
{
    "id": "q96",
    "question": "한국철도공사가 발주한 모바일오피스 시스템 고도화 용역에서 로그인 보안조치의 핵심 정책을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "5회",
        "SHA-256",
        "동시 로그인 차단"
    ],
    "expected_orgs": [
        "한국철도공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "로그인 실패 5회 시 접속 제한, 동시 로그인 차단, 일정 시간 미사용 시 자동 로그아웃, 사용자 비밀번호의 SHA-256 이상 일방향 암호화, 외부 접속계정의 이중인증 적용 및 내부 접속계정의 IP 또는 MAC 기반 접근을 요구합니다.",
    "expected_fields": {
        "발주기관": [
            "한국철도공사"
        ],
        "요구영역": [
            "로그인 보안조치",
            "사용자 인증"
        ]
    },
    "follow_up": {
        "question": "로그정보 저장 요구는 무엇이며 보관 기간은 얼마야?",
        "category": "follow_up",
        "expected_keywords": [
            "1년 이상",
            "로그정보",
            "암호화"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "모바일 단말기 접속 및 작업내역 등의 로그정보를 1년 이상 저장하고 암호화하여 위변조를 방지해야 합니다."
    }
} ,
{
    "id": "q97",
    "question": "한국철도공사 예약발매시스템 개량 ISMP 용역의 사업실적 평가 방식과 기본점수 부여 기준을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "기본점수 40%",
        "창업초기기업 50%",
        "최근 5년"
    ],
    "expected_orgs": [
        "한국철도공사"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 용역은 최근 5년 이내 동등이상과 유사 실적을 합산해 등급별 평점을 산정하고 기본점수는 배점한도의 40%(창업초기기업 50%)를 부여하며 가항목과 나항목 평점을 합산하되 총점은 10점을 초과할 수 없다.",
    "expected_fields": {
        "발주기관": [
            "한국철도공사"
        ],
        "요구영역": [
            "평가기준",
            "기본점수",
            "사업실적"
        ]
    },
    "follow_up": {
        "question": "기초금액 15억원에서 동등 8억원과 유사 4억원 실적일 때 평가점수는 얼마야.",
        "category": "follow_up",
        "expected_keywords": [
            "15억원",
            "4.20점",
            "2.16점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "평가점수는 10점으로 동등 4.20점과 유사 2.16점에 기본점수 4점을 합산한 값이다."
    }
} ,
{
    "id": "q98",
    "question": "한국한의학연구원이 발주한 통합정보시스템 고도화 용역의 기술평가 항목 구성과 주요 배점을 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "기술 및 기능 25점",
        "성능 및 품질 20점",
        "프로젝트 관리 20점"
    ],
    "expected_orgs": [
        "한국한의학연구원"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "본 용역의 기술평가는 기술 및 기능 25점, 성능 및 품질 20점, 프로젝트 관리 20점, 프로젝트 지원 10점 등으로 구성되며 각 항목은 기능요구 10점, 보안 5점, 데이터 5점, 시스템운영 3점, 제약 2점, 성능 10점, 품질 5점, 인터페이스 5점, 관리방법론 10점, 일정계획 5점, 개발장비 5점, 품질보증 2점, 시험운영 2점, 교육훈련 2점, 유지관리 1점, 하자보수 1점, 기밀보안 1점, 비상대책 1점의 세부 배점으로 평가됩니다.",
    "expected_fields": {
        "발주기관": [
            "한국한의학연구원"
        ],
        "요구영역": [
            "평가항목",
            "세부배점"
        ]
    },
    "follow_up": {
        "question": "표준 프레임워크 적용의 평가내용과 배점은 무엇이야?",
        "category": "follow_up",
        "expected_keywords": [
            "표준 프레임워크",
            "3점"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "표준 프레임워크 적용은 내용 이해도와 적용의 명확성 및 적용 방안의 타당성을 평가하며 배점은 3점입니다."
    }
} ,
{
    "id": "q99",
    "question": "개인정보 제공 및 활용 동의서의 수집 목적과 보유·이용 기간을 알려줘.",
    "category": "single_doc",
    "expected_keywords": [
        "용역입찰 제안서 심사평가",
        "성과 추적 완료"
    ],
    "expected_orgs": [
        "한국해양조사협회"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "개인정보 제공 및 활용 동의서의 수집 목적은 용역입찰 제안서 심사평가이고 보유·이용 기간은 용역 제안서 접수 시점부터 성과 추적이 완료되는 시점까지입니다.",
    "expected_fields": {
        "발주기관": [
            "한국해양조사협회"
        ],
        "요구영역": [
            "개인정보 수집목적",
            "보유기간"
        ]
    },
    "follow_up": {
        "question": "수집하려는 개인정보 항목은 무엇으로 기재되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "인적사항",
            "학력"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "수집하려는 개인정보 항목은 인적사항, 학력, 경력, 자격증 등입니다."
    }
} ,
{
    "id": "q100",
    "question": "한영대학교 특성화 맞춤형 교육환경 구축 사업의 유지관리 요구사항에서 무상하자보수 기간과 장애조치 의무를 요약해 줘.",
    "category": "single_doc",
    "expected_keywords": [
        "무상하자보수",
        "12개월"
    ],
    "expected_orgs": [
        "한영대학교"
    ],
    "expected_relevant_docs": 1,
    "reference_answer": "무상하자보수 기간은 검수일로부터 12개월이며 시스템 장애 발생 시 원인과 문제점을 파악하고 즉시 장애조치를 수행해야 합니다.",
    "expected_fields": {
        "발주기관": [
            "한영대학교"
        ],
        "요구영역": [
            "유지관리",
            "장애조치"
        ]
    },
    "follow_up": {
        "question": "유지관리 요구사항에서 신 버전의 소프트웨어가 출시될 경우 어떤 지원 협의를 하도록 되어 있어?",
        "category": "follow_up",
        "expected_keywords": [
            "신 버전 소프트웨어",
            "지원 협의"
        ],
        "expected_relevant_docs": 1,
        "reference_answer": "시스템 구축기간 및 하자보수 기간 중 신 버전의 소프트웨어가 출시될 경우 지원 협의를 하도록 명시되어 있습니다."
    }
} ,
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

