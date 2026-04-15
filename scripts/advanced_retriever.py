# 리랭킹
# 검색된 문서들의 순위를 재정렬하여 정답이 상단에 오게 만드는 로직
# Top-K 검색 결과에서 LLM이 각 문서의 관련성을 평가하여
# 최종적으로 가장 관련성이 높은 문서를 반환하는 방식입니다.

# 가상의 테스트 데이터베이스 (VectorStore) 생성
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document

# 업로드했던 raw 데이터 3개 중 일부 텍스트라고 가정합니다.
sample_docs = [
    Document(page_content="[개요] 본 사업은 고려대학교 차세대 포털 구축 사업입니다. 입찰 방법은 직찰입니다.", metadata={"source": "doc1.pdf"}),
    Document(page_content="[일정] 입찰서 접수 개시일은 2024년 5월 1일 10:00 이며, 제안서 제출 마감일은 2024년 5월 15일 14:00 입니다.", metadata={"source": "doc1.pdf"}),
    Document(page_content="[예산] 본 차세대 시스템 구축 사업의 배정예산은 부가가치세를 포함하여 총 1500000000원 입니다. 입찰한도액 동일.", metadata={"source": "doc1.pdf"}),
    Document(page_content="[기타] 본 입찰은 재공고 건입니다. 기존 최초 공고일은 2024년 4월 10일 이었습니다.", metadata={"source": "doc1.pdf"}),
    Document(page_content="소프트웨어 사업자 신고를 필한 업체만 참여 가능합니다.", metadata={"source": "doc1.pdf"})
]

# 임베딩 모델 로드 및 임시 벡터 DB 구축
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(sample_docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 기본 Top-K는 5로 설정

print("임시 벡터 DB 및 기본 검색기 생성 완료")





# Multi-Query Retriever 구현 및 테스트
import logging
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


# 로깅 설정을 통해 AI가 어떤 쿼리를 만들어내는지 터미널에서 확인 가능합니다.
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

query = "고려대 포털 사업의 총 예산과 입찰 일정을 알려줘."
print(f"** 원본 질문: {query}")
print("** 멀티 쿼리 검색 실행 중... (생성된 유사 질문들이 아래 출력됩니다) **")

# 멀티 쿼리를 통한 문서 검색
try:
    mq_docs = mq_retriever.invoke(query)
    print(f"\n** 멀티 쿼리 검색 결과: 총 {len(mq_docs)}개의 문서 조각을 찾았습니다.")
    for i, doc in enumerate(mq_docs):
        print(f"[{i+1}] {doc.page_content[:100]}...")
except Exception as e:
    print(f"멀티쿼리 실행 중 오류 발생: {e}")