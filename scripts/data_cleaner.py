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
        if not os.path.exists(self.csv_path): return
        df = pd.read_csv(self.csv_path)
        col = '파일명' if '파일명' in df.columns else 'file_name'
        
        if col in df.columns:
            initial_count = len(df)
            df_cleaned = df[~df[col].isin(self.discarded_files)]
            output_file = "data_list_cleaned.csv"
            df_cleaned.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"CSV 정제 완료: {initial_count}행 -> **{len(df_cleaned)}행**")
            print(f"저장된 파일명: {output_file}")
        else:
            print("CSV 컬럼 확인 불가")

if __name__ == "__main__":
    # 1. 원본 파일(PDF, HWP) 경로
    SERVER_RAW = "/srv/shared_data/pdf"
    LOCAL_RAW = "../data/files"
    RAW_PATH = SERVER_RAW if os.path.exists(SERVER_RAW) else LOCAL_RAW
    
    # 2. 읽어올 파일: AB님이 수동으로 고친 'fixed' 파일
    SERVER_CSV = "/srv/shared_data/datasets/data_list_fixed.csv"
    LOCAL_CSV = "../data/data_list_fixed.csv"
    CSV_PATH = SERVER_CSV if os.path.exists(SERVER_CSV) else LOCAL_CSV
    
    print(f"** 중복 제거 시작 **")
    print(f"참조 경로: {RAW_PATH}")
    
    matcher = SmartOriginFrequencyMatcher(RAW_PATH, CSV_PATH)
    matcher.run_process()
    
    # 참고: 결과물은 같은 폴더에 'data_list_cleaned.csv'로 저장됩니다.