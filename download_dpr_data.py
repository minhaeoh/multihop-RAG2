import os
import logging
from datasets import load_dataset
import requests
import gzip
from tqdm import tqdm

def download_file(url, local_path):
    """파일 다운로드 함수"""
    if os.path.exists(local_path):
        logging.info(f"File already exists: {local_path}")
        return
    
    logging.info(f"Downloading {url} to {local_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_dpr_data(data_dir="/home/minhae/multihop-RAG/wikipedia/dpr"):
    """Download and setup DPR data"""
    logging.info("Starting DPR data setup...")
    
    # 저장 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # 1. 위키피디아 원본 데이터 다운로드
        passages_path = os.path.join(data_dir, "psgs_w100.tsv.gz")
        if not os.path.exists(passages_path):
            download_file(
                "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
                passages_path
            )
        
        # 2. DPR 데이터셋과 FAISS 인덱스 다운로드
        logging.info("Loading wiki_dpr dataset...")
        wiki_dpr = load_dataset("facebook/wiki_dpr", 
                              name="psgs_w100.nq.exact",
                              with_embeddings=True,
                              with_index=True,
                              trust_remote_code=True)
        
        # FAISS 인덱스 저장
        index_path = os.path.join(data_dir, "psgs_w100.nq.exact.faiss")
        if not os.path.exists(index_path):
            logging.info("Saving FAISS index...")
            wiki_dpr['train'].save_faiss_index('embeddings', index_path)
        
        logging.info("DPR data setup completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Error in DPR data setup: {e}")
        raise

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # DPR 데이터 설정
    setup_dpr_data() 