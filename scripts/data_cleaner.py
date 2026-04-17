# 로직의 ##순서##를 변경해 데이터셋 정제 과정에서의 누락과
# 오류를 최소화하는 개선된 버전입니다.

import os
import hashlib
import re
import zlib
import pandas as pd
import csv
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader

try:
    import olefile
except ImportError:
    pass

class SmartOriginFrequencyMatcher:
    def __init__(self, raw_data_path, csv_path, fixed_csv_path):
        # 클래스 초기화 및 경로 설정
        self.raw_data_path = raw_data_path
        self.csv_path = csv_path
        self.fixed_csv_path = fixed_csv_path
        self.processed_records = {}
        self.discarded_files = set()

    def fix_summary_excel_error(self, text):
        # 엑셀 수식 오류 방지 처리
        if pd.isna(text) or str(text).strip() == "": return ""
        text = str(text).strip()
        if text.startswith(("=", "-", "+")):
            text = " " + text
        return text

    def clean_text_content(self, text):
        # 출력 가독성을 위한 최종 텍스트 정제('텍스트'컬럼용)
        if pd.isna(text) or str(text).strip() == "": return ""
        text = str(text)
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def parse_hwp(self, file_path):
        # HWP 파일 텍스트 추출
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

    def clean_text(self, text):
        # 스코어 계산용 텍스트 정제
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_content_hash(self, text):
        # 내용 기반 중복 식별용 해시 생성
        pure_text = re.sub(r"\s+", "", text)
        return hashlib.md5(pure_text.encode("utf-8")).hexdigest()

    def calculate_match_score(self, filename, content):
        # 기관명 빈도 기반 원본 판별 점수 계산(발주 기관은 다른데 내용이 동일한 경우를 구분하기 위함)
        org_name = filename.split('_')[0] if '_' in filename else filename[:10]
        org_name = re.sub(r'[^\w가-힣]', '', org_name)
        score = 0
        if org_name:
            count = content.count(org_name)
            score += count * 1000
            first_pos = content.find(org_name)
            if 0 <= first_pos < 500: score += 5000
            elif 500 <= first_pos < 1500: score += 2000
        file_path = os.path.join(self.raw_data_path, filename)
        score += os.path.getsize(file_path) / 1024 / 1024
        return score

    def run_process(self):
        # 파일 분석 및 중복 제거 실행
        if not os.path.exists(self.raw_data_path): return
        file_list = [f for f in os.listdir(self.raw_data_path) 
                     if f.lower().endswith((".pdf", ".hwp"))]
        
        print("** 1단계: 중복 분석 시작 **")
        for file_name in tqdm(file_list):
            file_path = os.path.join(self.raw_data_path, file_name)
            content = ""
            try:
                if file_name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    content = " ".join([d.page_content for d in loader.load()])
                elif file_name.lower().endswith(".hwp"):
                    content = self.parse_hwp(file_path)
                
                if not content or len(content.strip()) < 50: continue
                cleaned_text = self.clean_text(content)
                content_hash = self.get_content_hash(cleaned_text)
                current_score = self.calculate_match_score(file_name, cleaned_text)

                if content_hash in self.processed_records:
                    existing = self.processed_records[content_hash]
                    if current_score > existing['score']:
                        self.discarded_files.add(existing['filename'])
                        self.processed_records[content_hash] = {'filename': file_name, 'score': current_score}
                    else:
                        self.discarded_files.add(file_name)
                else:
                    self.processed_records[content_hash] = {'filename': file_name, 'score': current_score}
            except Exception as e:
                print(f"오류 ({file_name}): {e}")
        self.update_metadata_csv()

    def update_metadata_csv(self):
        # 보정 데이터 통합 및 컬럼 순서 유지 저장
        if not os.path.exists(self.csv_path): return
        
        # 1. 원본 로드 및 컬럼 순서 기억
        df = pd.read_csv(self.csv_path, dtype=str).fillna("")
        original_cols = df.columns.tolist() # 원본 순서 저장
        
        # 2. 중복 제거
        df_cleaned = df[~df['파일명'].isin(self.discarded_files)].copy()
        
        # 3. 보정 데이터 반영
        if os.path.exists(self.fixed_csv_path):
            print("** 2단계: 보정 데이터 반영 중 **")
            df_fixed = pd.read_csv(self.fixed_csv_path, dtype=str).fillna("")
            df_cleaned.set_index('파일명', inplace=True)
            df_fixed.set_index('파일명', inplace=True)
            
            target_cols = ['공고 번호', '사업 금액', '공개 일자', '입찰 참여 시작일', '입찰 참여 마감일']
            valid_cols = [c for c in target_cols if c in df_fixed.columns]
            
            df_cleaned.update(df_fixed[valid_cols])
            df_cleaned.reset_index(inplace=True)
            
            # 원본 컬럼 순서로 재배치
            df_cleaned = df_cleaned[original_cols]
        
        # 4. 최종 정제
        if '사업 요약' in df_cleaned.columns:
            df_cleaned['사업 요약'] = df_cleaned['사업 요약'].apply(self.fix_summary_excel_error)
        if '텍스트' in df_cleaned.columns:
            df_cleaned['텍스트'] = df_cleaned['텍스트'].apply(self.clean_text_content)

        # 원문 텍스트 계열 컬럼은 metadata 오염 방지를 위해 최종 CSV에서 제외
        excluded_text_cols = {"텍스트", "text", "content", "contents", "본문", "내용", "원문", "원문텍스트"}
        drop_cols = []
        for c in df_cleaned.columns:
            normalized = re.sub(r"[\s_]+", "", str(c).strip().lower())
            if (
                normalized in excluded_text_cols
                or normalized.endswith("text")
                or normalized.endswith("contents")
            ):
                drop_cols.append(c)
        if drop_cols:
            df_cleaned.drop(columns=drop_cols, inplace=True)
            print(f"** 원문 컬럼 제외: {', '.join(drop_cols)} **")

        # 입력 CSV와 같은 디렉토리에 cleaned 파일을 생성한다.
        output_dir = os.path.dirname(self.csv_path) if os.path.dirname(self.csv_path) else "."
        output_file = os.path.join(output_dir, "data_list_cleaned.csv")
        df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, escapechar='\\')
        print(f"\n** 작업 완료! 최종 파일: {output_file} **")

if __name__ == "__main__":
    # 1. 원본 파일(PDF, HWP) 경로 설정
    SERVER_RAW = "/srv/shared_data/pdf"
    LOCAL_RAW = "../data/files"
    RAW_PATH = SERVER_RAW if os.path.exists(SERVER_RAW) else LOCAL_RAW
    
    # 2. 원본 데이터 리스트 CSV 경로 설정
    SERVER_ORIGIN = "/srv/shared_data/datasets/data_list.csv"
    LOCAL_ORIGIN = "../data/data_list.csv"
    ORIGIN_PATH = SERVER_ORIGIN if os.path.exists(SERVER_ORIGIN) else LOCAL_ORIGIN
    
    # 3. 수동 보정본(fixed) CSV 경로 설정
    SERVER_FIXED = "/srv/shared_data/datasets/data_list_fixed.csv"
    LOCAL_FIXED = "../data/data_list_fixed.csv"
    FIXED_PATH = SERVER_FIXED if os.path.exists(SERVER_FIXED) else LOCAL_FIXED

    print(f"** 중복 제거 및 데이터 통합 프로세스 시작 **")
    print(f"참조 경로: {RAW_PATH}")
    print(f"보정 파일: {os.path.basename(FIXED_PATH)}")
    
    # 객체 생성 및 프로세스 실행
    matcher = SmartOriginFrequencyMatcher(RAW_PATH, ORIGIN_PATH, FIXED_PATH)
    matcher.run_process()
