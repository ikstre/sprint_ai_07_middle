# 기관명 빈도수 기반 중복 제거
# 수동 보정한 data_list_fixed.csv를 활용해
# 원본 데이터의 중복을 제거하고
# 최종 데이터셋(data_list_cleaned.csv)을 생성하는 핵심 도구입니다.

import os
import hashlib
import re
import zlib
import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

try:
    import olefile
except ImportError:
    pass

class SmartOriginFrequencyMatcher:
    def __init__(self, raw_data_path, csv_path):
        self.raw_data_path = raw_data_path
        self.csv_path = csv_path
        self.processed_records = {}    
        self.log_history = []          
        self.discarded_files = set()

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

    def clean_text(self, text):
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"[\t\r\n]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def get_content_hash(self, text):
        pure_text = re.sub(r"\s+", "", text)
        return hashlib.md5(pure_text.encode("utf-8")).hexdigest()

    def calculate_match_score(self, filename, content):
        # 1. 파일명에서 기관명 추출 (언더바 앞 단어)
        org_name = filename.split('_')[0] if '_' in filename else filename[:10]
        # 특수문자나 괄호가 포함되어 있다면 순수 텍스트만 추출
        org_name = re.sub(r'[^\w가-힣]', '', org_name)
        
        score = 0
        count = 0
        first_pos = -1
        
        if org_name:
            # 2. 본문 전체에서의 등장 빈도수 체크 (진짜 발주처는 수십 번 언급됨)
            count = content.count(org_name)
            score += count * 1000  # 1회 등장 시 1000점
            
            # 3. 최초 등장 위치 확인 (표지나 개요 등 최상단에 있을수록 높은 점수)
            first_pos = content.find(org_name)
            if 0 <= first_pos < 500:
                score += 5000  # 첫 500자 이내(표지) 등장 시 압도적 가점
            elif 500 <= first_pos < 1500:
                score += 2000  # 목차 수준에서 등장
                
        # 4. 파일 크기 보조 점수
        file_path = os.path.join(self.raw_data_path, filename)
        score += os.path.getsize(file_path) / 1024 / 1024
        
        return score, org_name, count, first_pos

    def run_process(self):
        file_list = [f for f in os.listdir(self.raw_data_path) 
                     if f.lower().endswith((".pdf", ".hwp"))]
        
        print(f"** 파일 분석 및 기관명 빈도수 대조 시작 (대상: {len(file_list)}개) **")
        
        for file_name in tqdm(file_list):
            file_path = os.path.join(self.raw_data_path, file_name)
            content = ""
            
            try:
                if file_name.lower().endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    content = " ".join([d.page_content for d in loader.load()])
                elif file_name.lower().endswith(".hwp"):
                    content = self.parse_hwp(file_path)
                
                if not content or len(content.strip()) < 50:
                    continue

                cleaned_text = self.clean_text(content)
                content_hash = self.get_content_hash(cleaned_text)
                
                # 빈도수와 위치 기반 원본성 점수 계산
                current_score, org_key, count, first_pos = self.calculate_match_score(file_name, cleaned_text)

                if content_hash in self.processed_records:
                    existing = self.processed_records[content_hash]
                    
                    if current_score > existing['score']:
                        # 현재 파일 점수가 더 높으면 교체 (올바른 발주처 파일 찾음)
                        self.log_history.append({
                            "type": "SWAP",
                            "kept": file_name,
                            "kept_org": org_key,
                            "kept_count": count,
                            "kept_score": current_score,
                            "discarded": existing['filename'],
                            "discarded_org": existing['org_key'],
                            "discarded_count": existing['count'],
                            "discarded_score": existing['score']
                        })
                        self.discarded_files.add(existing['filename'])
                        self.processed_records[content_hash] = {
                            'filename': file_name,
                            'score': current_score,
                            'org_key': org_key,
                            'count': count,
                            'text': cleaned_text
                        }
                    else:
                        # 기존 파일 점수가 더 높거나 같으면 현재 파일 제외
                        self.log_history.append({
                            "type": "DISCARD",
                            "kept": existing['filename'],
                            "kept_org": existing['org_key'],
                            "kept_count": existing['count'],
                            "kept_score": existing['score'],
                            "discarded": file_name,
                            "discarded_org": org_key,
                            "discarded_count": count,
                            "discarded_score": current_score
                        })
                        self.discarded_files.add(file_name)
                else:
                    self.processed_records[content_hash] = {
                        'filename': file_name,
                        'score': current_score,
                        'org_key': org_key,
                        'count': count,
                        'text': cleaned_text
                    }
                
            except Exception as e:
                print(f"파일 처리 중 오류 발생 ({file_name}): {e}")

        self.print_summary_log()
        self.update_metadata_csv()

    def print_summary_log(self):
        print("\n" + "="*90)
        print("내용 중복 및 원본 판별 상세 로그 (빈도 및 위치 기반)")
        print("="*90)
        
        if not self.log_history:
            print("중복된 문서가 발견되지 않았습니다.")
        else:
            for log in self.log_history:
                if log['type'] == "SWAP":
                    print(f"[교체] 보관: **{log['kept']}**")
                    print(f"      ㄴ 근거: 키워드({log['kept_org']}) 본문 {log['kept_count']}회 등장 (총점 {log['kept_score']:,.1f})")
                    print(f"      ㄴ 제외: {log['discarded']} (키워드 {log['discarded_org']} {log['discarded_count']}회 등장, 총점 {log['discarded_score']:,.1f})")
                    print("-" * 90)
                else:
                    print(f"[유지] 보관: **{log['kept']}**")
                    print(f"      ㄴ 근거: 키워드({log['kept_org']}) 본문 {log['kept_count']}회 등장 (총점 {log['kept_score']:,.1f})")
                    print(f"      ㄴ 제외: {log['discarded']} (키워드 {log['discarded_org']} {log['discarded_count']}회 등장, 총점 {log['discarded_score']:,.1f})")
                    print("-" * 90)
        
        print(f"최종 유니크 문서 확정: **{len(self.processed_records)}개**")
        print("="*90)

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