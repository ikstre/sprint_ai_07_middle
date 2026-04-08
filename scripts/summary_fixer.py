# 엑셀 오류(#NAME?)를 GPT로 자동 복구하는 기능입니다.
# 누락되거나 깨진 '사업 요약' 컬럼을
# AI가 원문을 보고 다시 작성해주는 복구 전용 스크립트입니다.

import os
import re
import zlib
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# LangChain 관련 모듈
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

try:
    import olefile
except ImportError:
    print("경고: olefile 라이브러리가 설치되지 않았습니다.")

# 환경변수 로드 (API 키 인식)
load_dotenv()

class SummaryFixer:
    def __init__(self, raw_data_path, target_csv):
        self.raw_data_path = raw_data_path
        self.target_csv = target_csv
        # 요약 작업은 정확도와 포맷 유지가 중요하므로 gpt-4o-mini 활용
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.setup_prompt()

    def setup_prompt(self):
        # 프롬프트 템플릿 구성
        template = """
제공된 공고문/제안요청서 텍스트를 바탕으로 핵심 내용을 요약하십시오.
작성 시 반드시 아래 제시된 항목과 하이픈(-) 불릿 양식을 엄격하게 유지해야 합니다.
내용이 부족한 항목은 본문에서 유추하여 간략히 채우되, 없는 내용은 생략 가능합니다.

[작성 양식]
- 사업개요: 
- 추진배경 및 필요성: 
- 사업내용(범위): 
- 기대효과: 
- 추진목표: 

문서 텍스트:
{context}
"""
        self.prompt = PromptTemplate.from_template(template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def parse_hwp(self, file_path):
        try:
            if not olefile.isOleFile(file_path): return ""
            f = olefile.OleFileIO(file_path)
            dirs = f.listdir()
            text_content = []
            for d in dirs:
                if d[0] == "BodyText":
                    stream = f.openstream("/".join(d))
                    data = stream.read()
                    try:
                        decoded_data = zlib.decompress(data, -15)
                        text_content.append(decoded_data.decode("utf-16", errors="ignore"))
                    except:
                        text_content.append(data.decode("utf-16", errors="ignore"))
            return " ".join(text_content)
        except: return ""

    def extract_text(self, file_path):
        # 문서의 핵심 개요는 보통 앞부분에 있으므로, 토큰 절약을 위해 상단 3000자만 추출
        content = ""
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                # 앞의 3장 정도만 읽어도 요약은 충분함
                docs = loader.load()[:3]
                content = " ".join([d.page_content for d in docs])
            elif file_path.lower().endswith(".hwp"):
                content = self.parse_hwp(file_path)
            
            # 태그 및 공백 정리
            content = re.sub(r"<[^>]*>", " ", content)
            content = re.sub(r"\s+", " ", content)
            return content[:3000] 
        except Exception as e:
            return ""

    def run_fix(self):
        if not os.path.exists(self.target_csv):
            print(f"오류: {self.target_csv} 파일을 찾을 수 없습니다.")
            return

        # 모든 데이터를 문자열로 로드 (지수 표기법 변형 방지)
        df = pd.read_csv(self.target_csv, dtype=str)

        # 1. 대상 컬럼 식별
        summary_col = None
        file_col = None
        
        for col in df.columns:
            if '요약' in col:
                summary_col = col
            if '파일명' in col or 'file' in col.lower():
                file_col = col

        if not summary_col or not file_col:
            print("오류: 요약 컬럼 또는 파일명 컬럼을 찾을 수 없습니다.")
            return

        # 2. #NAME? 오류가 있는 행 탐색
        # 엑셀 수식 오류는 보통 문자열 "#NAME?" 으로 저장됨
        error_mask = df[summary_col] == '#NAME?'
        target_rows = df[error_mask]
        
        target_count = len(target_rows)
        if target_count == 0:
            print("** 복구할 #NAME? 오류 셀이 없습니다. **")
            return
            
        print(f"** 총 {target_count}개의 #NAME? 오류 셀 복구 작업을 시작합니다. **")

        # 3. 데이터 복구 반복문
        success_count = 0
        for idx, row in tqdm(target_rows.iterrows(), total=target_count):
            file_name = str(row[file_col])
            file_path = os.path.join(self.raw_data_path, file_name)
            
            if not os.path.exists(file_path):
                print(f"파일 누락 건너뜀: {file_name}")
                continue
                
            # 텍스트 추출
            doc_text = self.extract_text(file_path)
            if len(doc_text) < 100:
                print(f"텍스트 추출 실패 건너뜀: {file_name}")
                continue
                
            # LLM을 통한 요약 생성
            try:
                generated_summary = self.chain.invoke({"context": doc_text})
                # 데이터프레임 업데이트
                df.at[idx, summary_col] = generated_summary
                success_count += 1
            except Exception as e:
                print(f"AI 요약 실패 ({file_name}): {e}")

        # 4. 결과 저장 (덮어씌우기)
        df.to_csv(self.target_csv, index=False, encoding="utf-8-sig")
        print(f"\n** 복구 완료: 총 {success_count}개의 셀이 AI로 새롭게 작성되었습니다. **")
        print(f"** 파일에 성공적으로 덮어씌웠습니다: {self.target_csv} **")

if __name__ == "__main__":
    # 1. 원본 파일 경로 (요약을 위해 텍스트를 읽어야 함)
    SERVER_RAW = "/srv/shared_data/pdf"
    LOCAL_RAW = "../data/files"
    RAW_PATH = SERVER_RAW if os.path.exists(SERVER_RAW) else LOCAL_RAW
    
    # 2. 수정할 대상: 에러가 있는 'fixed' 파일 그 자체 
    SERVER_TARGET = "/srv/shared_data/datasets/data_list_fixed.csv"
    LOCAL_TARGET = "../data/data_list_fixed.csv"
    TARGET_CSV = SERVER_TARGET if os.path.exists(SERVER_TARGET) else LOCAL_TARGET
    
    print(f"** AI 요약 복구 시작 **")
    print(f"대상 파일: {TARGET_CSV}")
    
    fixer = SummaryFixer(RAW_PATH, TARGET_CSV)
    fixer.run_fix()