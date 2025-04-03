import os
import argparse
import time
from datasets import load_dataset
from tqdm import tqdm
from multihop_solver import MultiHopSolver


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Multi-hop Solver for STEM questions')
    
    # Add arguments
    parser.add_argument('--model_id', 
                       type=str, 
                       default='meta-llama/Llama-3.3-70B-Instruct',
                       help='Model ID to use (default: meta-llama/Llama-3.3-70B-Instruct)')
    
    parser.add_argument('--ex_num', 
                       type=int,        
                       default=4,
                       help='Number of examples to generate (default: 4)')
    
    parser.add_argument('--subject', 
                       type=str, 
                       default='high_school_chemistry',
                       choices=['abstract_algebra', 'anatomy', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_physics', 'computer_security', 'conceptual_physics', 'electrical_engineering', 'elementary_mathematics', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'high_school_statistics', 'machine_learning'],
                       help='Subject to solve problems from (default: high_school_chemistry)')
    
    parser.add_argument('--generate_ex', 
                       type=str2bool, 
                       default=True,
                       help='Whether to generate example problems (default: True)')
    
    parser.add_argument('--review_ex', 
                       type=str2bool, 
                       default=False,
                       help='Whether to review examples (default: True)')
    
    parser.add_argument('--review_doc', 
                       type=str2bool, 
                       default=False,
                       help='Whether to review documents (default: True)')
    
    parser.add_argument('--zeroshot', 
                       type=str2bool, 
                       default=False,
                       help='Whether to use zero-shot approach (default: False)')
    
    parser.add_argument('--summarize', 
                       type=str2bool, 
                       default=False,
                       help='Whether to summarize retrieved documents (default: False)')
    
    parser.add_argument('--contents_path',
                       type=str,
                       default="/home/minhae/multihop-RAG/wikipedia/dpr/wiki_contents.json",
                       help='Path to wiki contents file')
    
    parser.add_argument('--chunk_path',
                       type=str,
                       default="/home/minhae/multihop-RAG/wikipedia/dpr/wiki_chunks.json",
                       help='Path to wiki chunks file')
    
    parser.add_argument('--embeddings_path',
                       type=str,
                       default="/home/minhae/multihop-RAG/wikipedia/dpr/embeddings.npy",
                       help='Path to embeddings file')
    
    parser.add_argument('--index_path',
                       type=str,
                       default="/home/minhae/multihop-RAG/wikipedia/dpr/wikipedia_index.faiss",
                       help='Path to FAISS index file')
    
    # Parse arguments
    args = parser.parse_args()
    
    
    # Load dataset to get total number of problems for the subject
    ds = load_dataset("TIGER-Lab/MMLU-STEM")
    data = ds['test'].filter(lambda x: x['subject'].lower()==args.subject)
    total_problems = len(data)
    
    print(f"Running all {total_problems} problems for {args.subject}")
    
    # Initialize solver
    solver = MultiHopSolver(
        model_id=args.model_id,
        ex_num=args.ex_num,
        subject=args.subject,
        generate_ex=args.generate_ex,
        review_ex=args.review_ex,
        review_doc=args.review_doc,
        zeroshot=args.zeroshot,
        summarize=args.summarize
    )
    
    # 각 문제를 순차적으로 처리
    correct_count = 0
    total_count = 0
    for prob_num in tqdm(range(1, total_problems + 1), desc="Processing problems"):
        try:
            solver.solve_problem(args.subject, prob_num)
            # 필요한 경우 여기서 결과 처리 및 통계 계산
            total_count += 1
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"Error processing problem {prob_num}: {e}")

if __name__ == "__main__":
    main()