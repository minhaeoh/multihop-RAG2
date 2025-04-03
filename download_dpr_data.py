import os
import logging
from datasets import load_dataset
import requests
import gzip
from tqdm import tqdm

def download_file(url, local_path):
    """File download function"""
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

def setup_dpr_data(data_dir=None):
    """Download and setup DPR data"""
    if data_dir is None:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create path relative to the script location
        data_dir = os.path.join(script_dir, "wikipedia", "dpr")
    
    logging.info("Starting DPR data setup...")
    logging.info(f"Data will be stored in: {data_dir}")
    
    # Create storage directory
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # 1. Download Wikipedia original data
        passages_path = os.path.join(data_dir, "psgs_w100.tsv.gz")
        if not os.path.exists(passages_path):
            download_file(
                "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
                passages_path
            )
        
        # 2. Download DPR dataset and FAISS index
        logging.info("Loading wiki_dpr dataset...")
        wiki_dpr = load_dataset("facebook/wiki_dpr", 
                              name="psgs_w100.nq.exact",
                              with_embeddings=True,
                              with_index=True,
                              trust_remote_code=True)
        
        # Save FAISS index
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
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup DPR data
    setup_dpr_data() 