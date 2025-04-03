import os
import re
import time
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DPRQuestionEncoderTokenizer, 
    DPRQuestionEncoder
)
import faiss
import gzip
import csv
import wandb

class MultiHopSolver:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct", ex_num=2, subject="high_school_chemistry", generate_ex=True, review_doc=False, review_ex=False, zeroshot=False, summarize=False):
        # Initialize environment
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(dotenv_path=env_path)
        
        self.model_id = model_id
        self.ex_num = ex_num
        self.subject = subject
        self.generate_ex = generate_ex
        self.review_doc = review_doc
        self.review_ex = review_ex
        self.zeroshot = zeroshot
        self.summarize = summarize
        
        # Initialize timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directory with all argument information
        base_dir = os.path.join(os.path.dirname(__file__), "results")
        dir_name = f"{self.subject}_{self.model_id.split('/')[-1]}"
        
        # Add configuration details to directory name
        config_parts = []
        if self.zeroshot:
            config_parts.append("zeroshot")
        if self.generate_ex:
            config_parts.append(f"ex{self.ex_num}")
        else:
            config_parts.append("no_example")
            
        if self.review_doc:
            config_parts.append("doc_review")
        if self.review_ex:
            config_parts.append("ex_review")
            
        config_str = "_".join(config_parts)
        self.output_dir = os.path.join(base_dir, f"{dir_name}_{config_str}", self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set device for CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize the Llama model 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.model.eval()
            print("Successfully initialized Llama model")
        except Exception as e:
            print(f"Error initializing Llama model: {e}")
            raise
        

        print("Setting up DPR components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Query encoder & tokenizer 초기화
        try:
            self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.q_encoder = self.q_encoder.to(self.device)
            self.q_encoder.eval()
        except Exception as e:
            print(f"Error initializing DPR components: {e}")
            raise
         
        # 2. FAISS 인덱스와 패시지 데이터 로드
        data_dir = os.path.join(os.path.dirname(__file__), "wikipedia", "dpr")
        try:
            # FAISS 인덱스 로드
            index_path = os.path.join(data_dir, "psgs_w100.nq.exact.faiss")
            self.index = faiss.read_index(index_path)
            
            # 원본 텍스트 데이터 로드
            passages_path = os.path.join(data_dir, "psgs_w100.tsv.gz")
            self.passages = []
            with gzip.open(passages_path, 'rt', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)  # 헤더 스킵
                for row in reader:
                    if len(row) >= 3:  # id, text, title
                        self.passages.append({
                            'id': row[0],
                            'text': row[1][1:-1].replace('""', '"'),  # 따옴표 처리
                            'title': row[2]
                        })
        except Exception as e:
            print(f"Error loading FAISS index or passages: {e}")
            raise
        
        # Initialize wandb
        self.run = wandb.init(
            project=f"multihop-RAG-{self.subject}",
            entity="minhae",
            config={
                "model_id": model_id,
                "ex_num": ex_num,
                "subject": subject,
                "generate_ex": generate_ex,
                "review_doc": review_doc,
                "review_ex": review_ex,
                "zeroshot": zeroshot,
                "summarize": summarize
            }
        )
        self.run.name = f"{dir_name}_{config_str}"

    def generate_text(self, prompt, strip_prompt, max_length=512, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
        """Generate text using the Llama pipeline with output cleaning"""
        try:
            
            stop_phrases = [
                "Let me",
                "I hope",
                "Please",
                "Here's",
                "I'll",
                "I can",
                "Feel free",
                "Do you",
                "Note:",
                "Remember:",
                "Best regards",
                "Best,",
                "(Note:",
                "Thank you",
                "I am happy",
                "def ",
                "import ",
            ]
            bad_words_ids = [self.tokenizer.encode(phrase, add_special_tokens=False) for phrase in stop_phrases]
            
            # Encode the input with proper attention mask
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                bad_words_ids=bad_words_ids
            )
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = output_text[len(strip_prompt):].strip()
            response = re.sub(r"(\|\s*){3,}", "", response).strip()
            
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

    def generate_subquestions(self, problem):
        formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""Question: {problem['question']}
Options: {choices_text}

Break this problem into maximum 5 essential subquestions that directly help solve the original problem.
Each subquestion MUST include its solution and a search query.

STRICT FORMAT REQUIREMENTS:
1. For each subquestion, you MUST provide exactly three parts in this order:
   - The subquestion
   - The solution to that subquestion
   - A search query for that subquestion

2. Use EXACTLY this format for each subquestion:
Subquestion 1: [your specific subquestion]
Solution for Subquestion 1: [your specific solution]
Search Query for Subquestion 1: [your specific search query]
"""
        prompt = base_prompt + "\n\nSubquestion 1: "

        # Get raw response and clean it
        raw_response = self.generate_text(prompt,base_prompt)
        
        # Split into lines and find where "Final Answer" appears
        lines = raw_response.split('\n')
        final_lines = []
        current_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Subquestion") or line.strip().startswith("Solution for Subquestion") or line.strip().startswith("Search Query for Subquestion"):
                current_index = i
            elif "final answer" in line.lower() or "Answer:" in line or "Answer :" in line:
                current_index = i+1
                break
            elif line.strip() == "":
                current_index = i
            else: break

        final_lines = lines[:current_index]  # Keep only lines before "Final Answer"
                
        # Join the lines back together
        cleaned_response = '\n'.join(final_lines)
        
        return cleaned_response

    def review_document(self, query, document):
        """Review if a document is relevant to the query using Llama"""
        base_prompt = f"""You are a document reviewer. Your task is to determine if the given document is relevant to answering the query.

Query: {query}

Document: {document}

Please analyze if this document is relevant to answering the query. Consider:
1. Does the document contain information directly related to the query?
2. Is the information in the document accurate and up-to-date?
3. Is the document's content specific enough to be useful?

Respond with only "RELEVANT" or "NOT_RELEVANT"."""
        prompt = base_prompt + "\nThis document is"

        try:
            response = self.generate_text(prompt,base_prompt, max_length=50)
            result = response.strip().upper()
            if 'RELEVANT' in result:
                return True
            else:
                return False
        except Exception as e:
            print(f"Error in document review: {e}")
            return False

    def summarize_document(self, document):
        """Summarize a document using Llama model"""
        base_prompt = f"""Please provide a concise summary of the following document. Focus on the key information and main points.

Document:
{document}

Summary:"""

        try:
            summary = self.generate_text(base_prompt, base_prompt, max_length=200)
            return summary.strip()
        except Exception as e:
            print(f"Error in document summarization: {e}")
            return document

    def get_wiki_search_results(self, query, num_results=3, review=True):
        """Get search results using DPR and FAISS"""
        try:
            # 쿼리 인코딩
            with torch.no_grad():
                query_inputs = self.q_tokenizer(
                    [query],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                query_vector = self.q_encoder(**query_inputs).pooler_output
                query_vector = query_vector.cpu().numpy()
            
            # FAISS로 검색
            search_num = num_results * 2 if review else num_results
            distances, indices = self.index.search(query_vector, search_num)
            
            documents = []
            for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= len(self.passages):
                    continue
                    
                passage = self.passages[idx]
                content = passage['text']
                title = passage['title']
                
                # 문서 요약 (옵션)
                if self.summarize:
                    content = self.summarize_document(content)
                
                doc = f"Document {i+1} (score: {score:.2f}) [Title: {title}]:\n{content}"
                
                # 문서 관련성 검토 (옵션)
                if review:
                    print(f"Reviewing document {i+1}/{search_num}...")
                    if self.review_document(query, content):
                        documents.append(doc)
                        if len(documents) >= num_results:
                            break
                else:
                    documents.append(doc)
                    if len(documents) >= num_results:
                        break
            
            if not documents:
                return "No relevant documents found."
            
            return "\n\n".join(documents)
            
        except Exception as e:
            print(f"Error in wiki search: {e}")
            return "No relevant documents found."

    def review_example(self, subquestion, example):
        """Review if a generated example is relevant and helpful for understanding the subquestion"""
        base_prompt = f"""You are an example reviewer. Your task is to determine if the given example is relevant and helpful for understanding the subquestion.

Subquestion: {subquestion}

Example:
{example}

Please analyze if this example is relevant and helpful. Consider:
1. Does the example directly relate to the concept in the subquestion?
2. Is the example clear and well-structured?
3. Does the example help in understanding the reasoning needed for the subquestion?
4. Is the solution detailed and instructive?

Respond with only "RELEVANT" or "NOT_RELEVANT"."""
        prompt = base_prompt + "\nThis example is"


        try:
            response = self.generate_text(prompt, base_prompt, max_length=50)
            result = response.strip().upper()
            return result == "RELEVANT"
        except Exception as e:
            print(f"Error in example review: {e}")
            return False

    def generate_examples(self, problem, documents, subquestions, step_num):
        """Step 2: Generate example questions for each subquestion"""
        formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        # Adjust number of examples to generate based on whether we'll review them
        num_to_generate = self.ex_num * 2 if self.review_ex else self.ex_num

        base_prompt = f"""I am trying to solve a complex question by breaking it down step by step. For a given subquestion, I want example questions with full solutions that help understand the concept involved in that step. You will be given the original question, a subquestion, and a knowledge document.

        Your task is:
        - Generate {num_to_generate} example questions with full solutions based *only* on the knowledge document.
        - The questions and solutions must rely strictly on the given document. Do not use outside knowledge.
        - The questions should help understand the concept mentioned in the subquestion.
        - Do not solve the original question.
        - Do not use multiple choice or ask the user to select an option. Make each question open-ended or short-answer style.
        - Do not mention or refer to the document in the questions or solutions. The outputs should be self-contained.
        - Each solution should be accurate, clear, and informative.

        Format:
        Question 1: [Write the question here]
        Solution 1: [Write the full solution here]

        ...

        Here is the input:
        Question: {problem['question']} | Options: {choices_text}
        Stepquestion {step_num}: {subquestions[step_num-1]}
        Doc: {documents}"""
        prompt = base_prompt + "\n\nQuestion 1:"

        response = self.generate_text(prompt,base_prompt)
        
        # Parse the generated examples
        raw_examples = re.split(r"```+", response)[0]
        examples_list = []
        current_example = ""
        
        # Split the response into individual examples
        for line in raw_examples.split('\n'):
            if line.strip().startswith("Solution 4:"):
                current_example += '\n' + line
                examples_list.append(current_example.strip())
                break  # 이후 내용 무시
            elif line.strip().startswith("Question"):
                if current_example:
                    examples_list.append(current_example.strip())
                current_example = line
            elif line.strip():
                current_example += '\n' + line

        if current_example and not any("Solution 4:" in ex for ex in examples_list):
            examples_list.append(current_example.strip())
        
        if self.review_ex:
            print(f"\nReviewing {len(examples_list)} examples...")
            # Review and filter examples
            relevant_examples = []
            for i, example in enumerate(examples_list, 1):
                print(f"Reviewing example {i}/{len(examples_list)}...")
                if self.review_example(subquestions[step_num-1], example):
                    relevant_examples.append(example)
                    if len(relevant_examples) >= self.ex_num:
                        break
            print(f"Found {len(relevant_examples)} relevant examples")
            return "\n\n".join(relevant_examples)
        else:
            #print("\nReview example error")
            # Return all examples without review
            return "\n\n".join(examples_list)

    def solve_subquestion(self, problem, subquestions, examples, subsolutions, step_num):
        """Steps 3-5: Solve each subquestion"""
        formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""You are solving a complex problem step by step. You will be given:

            1. The original question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Example problems with solutions that demonstrate how to solve the current subquestion

            Your task:
            - Carefully read the original question, any previous solutions, and the current subquestion
            - Use the reasoning and methods shown in the example problems to solve the current subquestion
            - Your solution should be detailed and logically structured

            Question: {problem['question']} | Options: {choices_text}"""
        
        # Add previous subquestions and their solutions if they exist
        if step_num > 1:
            for i in range(step_num - 1):
                base_prompt += f"""
                
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}"""
        
        # Add current subquestion and examples
        base_prompt += f"""
                
                Subquestion {step_num}: {subquestions[step_num-1]}
                
                Example Problems:
                {examples}
                
                Now write the Step {step_num} Solution.
                """
        prompt = base_prompt + f"\n\nStep {step_num} Solution:"

        response = self.generate_text(prompt,base_prompt)
        solutions = []

        for line in response.split("\n"):
            if 'final answer' in line.lower():
                solutions.append(line)
                break
            else: 
                solutions.append(line)

        return "\n".join(solutions)

    def solve_subquestion_with_docs(self, problem, subquestions, documents, subsolutions, step_num):
        """Steps 3-5: Solve each subquestion using retrieved documents directly"""
        formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""You are solving a complex problem step by step. You will be given:

            1. The original question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Retrieved documents containing relevant information

            Your task:
            - Carefully read the original question, any previous solutions, and the current subquestion
            - Use the information from the retrieved documents to solve the current subquestion
            - Your solution should be detailed and logically structured

            Question: {problem['question']} | Options: {choices_text}"""
        
        # Add previous subquestions and their solutions if they exist
        if step_num > 1:
            for i in range(step_num - 1):
                base_prompt += f"""
                
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}"""
        
        # Add current subquestion and documents
        base_prompt += f"""
                
                Subquestion {step_num}: {subquestions[step_num-1]}
                
                Retrieved Documents:
                {documents}
                
                [Now write the Step {step_num} Solution:]
                """
        response = self.generate_text(base_prompt,base_prompt)
        responses = response.split("\n")
        index = 0
        for i, line in enumerate(responses):
            if 'final answer' in line.lower():
                index = min(i+3, len(responses))
                break
            else: index = i

        return "\n".join(responses[:index])

    def generate_final_answer(self, problem, subquestions, subsolutions):
        """Step 6: Generate final answer using all subquestion solutions"""
        formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(problem['choices'])]
        choices_text = "\n".join(formatted_choices)
        base_prompt = f"""
            You are solving a complex problem that has been broken into four subquestions. Each subquestion has already been solved. Your task is to carefully read the original question and the four subquestion solutions, then use them to determine the final answer.

            Instructions:
            - You will be given the original question and the several subquestion solutions.
            - Read the original question carefully.
            - Review each subquestion and its solution.
            - Use the information and reasoning from all four subsolutions to logically determine the final answer.
            - Make sure your answer is consistent with the subsolutions.
            - Your output should include a brief justification followed by the final answer.

            Output format:

            Final Reasoning:
            [Explain how the subsolutions lead to the final answer.]

            Final Answer:
            [ONLY the number (0, 1, 2, or 3) of the correct option]

            Here is the input:

            Question: {problem['question']} | Options: {choices_text}

            """
        
        # Add all subquestions and their solutions
        for i, (subq, sol) in enumerate(zip(subquestions, subsolutions), 1):
            base_prompt += f"\nSubquestion {i}: {subq}\nSubquestion {i} Solution: {sol}\n"
        prompt = base_prompt + "\n\nFinal Reasoning:"

        response = self.generate_text(prompt,base_prompt)
        print(response)
        responses = response.split("\n")
        index = 0
        for i, line in enumerate(responses):
            if 'final answer' in line.lower():
                index = min(i+3, len(responses))
                break
            else: index = i
        
        return "\n".join(responses[:index])

    def parse_subquestions_and_queries(self, result_text):
        """Parse subquestions, solutions, and search queries from the generated text"""
        subquestions = []
        solutions = []
        search_queries = []
        
        # Split the text into lines and process each line
        lines = result_text.strip().split('\n')
        
        current_subq = None
        current_sol = None
        current_query = None
        current_subq_num = None
        
        #print("Processing lines:")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            #print(f"Processing line: {line}")
            
            # Extract subquestion number if present
            subq_match = re.match(r"Subquestion (\d+):", line)
            if subq_match and "Solution" not in line:
                # If we have a previous incomplete set with subquestion and query
                if current_subq and current_query and current_subq_num:
                    print(f"Adding set with missing solution: {current_subq} | {current_query}")
                    subquestions.append(current_subq)
                    solutions.append("")  # Empty solution
                    search_queries.append(current_query)
                
                current_subq_num = int(subq_match.group(1))
                current_subq = line.split(":", 1)[1].strip()
                current_sol = None
                current_query = None
                
            # Process solution if it matches current subquestion number
            elif "Solution:" in line:
                sol_match = re.match(r"Solution for Subquestion (\d+):", line)
                if sol_match and int(sol_match.group(1)) == current_subq_num:
                    current_sol = line.split("Solution:", 1)[1].strip()
                
            # Process search query if it matches current subquestion number
            elif line.startswith("Search Query for Subquestion"):
                query_match = re.match(r"Search Query for Subquestion (\d+):", line)
                if query_match and int(query_match.group(1)) == current_subq_num:
                    current_query = line.split(":", 1)[1].strip()
                
                    # If we have both subquestion and query (solution optional)
                    if current_subq and current_query:
                        print(f"Adding complete set: {current_subq} | {current_sol or ''} | {current_query}")
                        subquestions.append(current_subq)
                        solutions.append(current_sol or "")  # Empty string if no solution
                        search_queries.append(current_query)
                    
                        # Reset for next set
                        current_subq = None
                        current_sol = None
                        current_query = None
                        current_subq_num = None
        
        # Handle the last set if incomplete
        if current_subq and current_query:
            print(f"Adding final set: {current_subq} | {current_sol or ''} | {current_query}")
            subquestions.append(current_subq)
            solutions.append(current_sol or "")
            search_queries.append(current_query)

        print(f"\nFinal counts - Subquestions: {len(subquestions)}, Solutions: {len(solutions)}, Queries: {len(search_queries)}")
        
        # Verify that we have equal numbers of subquestions and queries
        if not (len(subquestions) == len(search_queries)):
            raise ValueError("Parsing error: Mismatch in number of subquestions and queries")
        
        return subquestions, solutions, search_queries

    def solve_with_zeroshot(self, problem):
        """Solve the problem directly without generating subquestions or using RAG"""
        formatted_choices = [f"{i}: {choice}" for i, choice in enumerate(problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""Solve this multiple choice question step by step.
Question: {problem['question']} | Options: {choices_text}

Output format:

Final Reasoning:
[Explain how the subsolutions lead to the final answer.]

Final Answer:
[ONLY the number (0, 1, 2, or 3) of the correct option]

"""
        prompt = base_prompt + "\n\nFinal Reasoning:"

        try:
            response = self.generate_text(prompt,base_prompt)
            print(response)
            responses = response.split("\n")
            index = 0
            for i, line in enumerate(responses):
                if 'final answer' in line.lower():
                    index = min(i+3, len(responses))
                    break
                else: index = i

            return "\n".join(responses[:index])
        except Exception as e:
            print(f"Error in zeroshot solution: {e}")
            return None

    def solve_problem(self, subject, prob_num):
        """Main method to solve the problem end-to-end"""
        # Add timing dictionary
        timing_stats = {}
        start_total = time.time()

        # Load dataset
        start = time.time()
        ds = load_dataset("TIGER-Lab/MMLU-STEM")
        data = ds['test'].filter(lambda x: x['subject'].lower()==subject)
        problem = data[prob_num - 1]
        timing_stats['data_loading'] = time.time() - start

        # Create output files
        detailed_file = os.path.join(self.output_dir, f"{self.timestamp}_{subject}_detailed.txt")
        summary_file = os.path.join(self.output_dir, f"{self.timestamp}_{subject}_summary.txt")

        def file_print(*args, **kwargs):
            with open(detailed_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)

        def summary_print(*args, **kwargs):
            with open(summary_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)
                
        try:
            # Print configuration information only for the first problem
            if prob_num == 1:
                summary_print("Configuration Information:")
                summary_print("=" * 50)
                summary_print(f"Model ID: {self.model_id}")
                summary_print(f"Subject: {self.subject}")
                summary_print(f"Number of Examples: {self.ex_num}")
                summary_print(f"Generate Examples: {self.generate_ex}")
                summary_print(f"Review Documents: {self.review_doc}")
                summary_print(f"Review Examples: {self.review_ex}")
                summary_print(f"Device: {self.device}")
                summary_print(f"Timestamp: {self.timestamp}")
                summary_print("=" * 50)
                summary_print("\nProblem-by-Problem Results:")
                summary_print(f"{'Problem':^10} | {'Result':^10} | {'Predicted':^10} | {'Actual':^10} | {'Running Acc':^12}")
                summary_print("-" * 60)

            # Print problem and options to both files
            for print_func in [file_print, summary_print]:
                print_func("\nProblem {}:".format(prob_num))
                print_func("Question: {}".format(problem['question']))
                print_func("Choices: {}".format(problem['choices']))
                print_func("Correct Answer: {}".format(problem['answer']))
                print_func("--------------------------------")

            if self.zeroshot:
                start = time.time()
                file_print("Solving with zero-shot approach...")
                solution = self.solve_with_zeroshot(problem)
                timing_stats['zeroshot_solution'] = time.time() - start
                
                file_print("\nZero-shot Solution:")
                file_print(solution)
                
                # Extract predicted answer - look for "final answer" in any case
                predicted_answer = None
                solution_lines = solution.split('\n')
                for i, line in enumerate(solution_lines):
                    if 'final answer' in line.lower():
                        try:
                            answer_line = ''.join(solution_lines[i:])
                            # Get the first number after "Final Answer:"
                            numbers = re.findall(r'\d+', answer_line.split(':', 1)[1])
                            if numbers:
                                predicted_answer = int(numbers[0])
                                break
                        except:
                            continue
                if predicted_answer is None:
                    predicted_answer = 5
                
                # Update running accuracy statistics
                if hasattr(self, '_total_problems'):
                    self._total_problems += 1
                else:
                    self._total_problems = 1
                
                if hasattr(self, '_correct_problems'):
                    if predicted_answer == problem['answer']:
                        self._correct_problems += 1
                else:
                    self._correct_problems = 1 if predicted_answer == problem['answer'] else 0
                
                current_accuracy = (self._correct_problems / self._total_problems) * 100
                
                # Log metrics to wandb
                self.run.log({
                    "accuracy": current_accuracy,
                    "correct_problems": self._correct_problems,
                    "total_problems": self._total_problems,
                    "problem_num": prob_num,
                    "predicted_answer": predicted_answer,
                    "actual_answer": problem['answer'],
                })
                
                # Save to summary file with running accuracy
                if predicted_answer is not None:
                    result = "Correct" if predicted_answer == problem['answer'] else "Wrong"
                    summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {problem['answer']:^10} | {current_accuracy:^11.2f}%")
                    
                    # Save detailed results to the detailed file
                    file_print("\nFinal Results:")
                    file_print(f"Predicted Answer: {predicted_answer}")
                    file_print(f"Actual Answer: {problem['answer']}")
                    file_print(f"Result: {result}")
                    file_print(f"Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
                
                file_print("\nTiming Statistics:")
                file_print("=" * 50)
                file_print(f"Zero-shot Solution: {timing_stats['zeroshot_solution']:.2f} seconds")

            else:
                # Step 1: Generate subquestions
                start = time.time()
                file_print("Step 1: Generating subquestions and search queries...")
                subq_result = self.generate_subquestions(problem)
                file_print(subq_result)
                timing_stats['generate_subquestions'] = time.time() - start
                
                # Parse the results
                start = time.time()
                subquestions, initial_solutions, search_queries = self.parse_subquestions_and_queries(subq_result)
                timing_stats['parse_subquestions'] = time.time() - start
                
                file_print("There are ", len(subquestions), " subquestions")
                
                # Initialize storage for final solutions
                subsolutions = []
                
                # Steps 2-5: Generate examples and solve each subquestion
                timing_stats['subquestions'] = {}
                for step_num in range(1, len(subquestions) + 1):
                    step_timing = {}
                    file_print(f"\nStep {step_num + 1}: Processing subquestion {step_num}...")
                    
                    # Time document retrieval
                    start = time.time()
                    documents = self.get_wiki_search_results(search_queries[step_num - 1], review=self.review_doc)
                    step_timing['document_retrieval'] = time.time() - start

                    if documents == "No relevant documents found.":
                        documents = ""
                        file_print(f"\nNo relevant documents found for Subquestion {step_num}.")
                    else:
                        # Save search results
                        file_print(f"\nSearch Results for Subquestion {step_num}:")
                        file_print(documents)
                    
                    if self.generate_ex:
                        # Time example generation
                        start = time.time()
                        examples = self.generate_examples(problem, documents, subquestions, step_num)
                        step_timing['example_generation'] = time.time() - start
                        
                        file_print(f"\nGenerated Examples for Subquestion {step_num}:")
                        file_print(examples)
                        
                        # Time solution generation with examples
                        start = time.time()
                        file_print("Solving subquestion with examples...")
                        solution = self.solve_subquestion(problem, subquestions, examples, 
                                                       subsolutions, step_num)
                        step_timing['solution_generation'] = time.time() - start
                    else:
                        # Time direct solution generation
                        start = time.time()
                        solution = self.solve_subquestion_with_docs(problem, subquestions, 
                                                                  documents, subsolutions, step_num)
                        step_timing['solution_generation'] = time.time() - start
                    
                    timing_stats['subquestions'][f'step_{step_num}'] = step_timing
                    subsolutions.append(solution)
                    
                    file_print(f"\nSolution for subquestion {step_num}:")
                    file_print(solution)
                    
                    file_print("\n" + "="*50 + "\n")  # 구분선 추가
                
                # Step 6: Generate final answer
                start = time.time()
                file_print("\nStep 6: Generating final answer...")
                final_answer = self.generate_final_answer(problem, subquestions, subsolutions)
                file_print(final_answer)
                timing_stats['final_answer'] = time.time() - start

                # Extract predicted answer and update accuracy
                predicted_answer = None
                final_answer_lines = final_answer.split('\n')
                for i, line in enumerate(final_answer_lines):
                    if 'final answer' in line.lower():
                        try:    
                            answer_line = ''.join(final_answer_lines[i:])
                            # Get the first number after "Final Answer:"
                            numbers = re.findall(r'\d+', answer_line.split(':', 1)[1])
                            if numbers:
                                predicted_answer = int(numbers[0])
                                break
                        except:
                            continue
                if predicted_answer is None:
                    predicted_answer = 5
                
                # Update running accuracy statistics
                if hasattr(self, '_total_problems'):
                    self._total_problems += 1
                else:
                    self._total_problems = 1
                
                if hasattr(self, '_correct_problems'):
                    if predicted_answer == problem['answer']:
                        self._correct_problems += 1
                else:
                    self._correct_problems = 1 if predicted_answer == problem['answer'] else 0
                
                current_accuracy = (self._correct_problems / self._total_problems) * 100
                
                # Log metrics to wandb
                self.run.log({
                    "accuracy": current_accuracy,
                    "correct_problems": self._correct_problems,
                    "total_problems": self._total_problems,
                    "problem_num": prob_num,
                    "predicted_answer": predicted_answer,
                    "actual_answer": problem['answer'],
                    "is_correct": predicted_answer == problem['answer']
                })

                # Save to summary file with running accuracy
                if predicted_answer is not None:
                    result = "Correct" if predicted_answer == problem['answer'] else "Wrong"
                    summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {problem['answer']:^10} | {current_accuracy:^11.2f}%")
                    
                    # Save detailed results to the detailed file
                    file_print("\nFinal Results:")
                    file_print(f"Predicted Answer: {predicted_answer}")
                    file_print(f"Actual Answer: {problem['answer']}")
                    file_print(f"Result: {result}")
                    file_print(f"Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
                
                # Print timing statistics
                timing_stats['total_time'] = time.time() - start_total
                file_print("\nTiming Statistics:")
                file_print("=" * 50)
                file_print(f"Total Time: {timing_stats['total_time']:.2f} seconds")
                file_print(f"Data Loading: {timing_stats['data_loading']:.2f} seconds")
                file_print(f"Generate Subquestions: {timing_stats['generate_subquestions']:.2f} seconds")
                file_print(f"Parse Subquestions: {timing_stats['parse_subquestions']:.2f} seconds")
                
                for step_num, step_timing in timing_stats['subquestions'].items():
                    file_print(f"\n{step_num.upper()}:")
                    for operation, duration in step_timing.items():
                        file_print(f"  {operation}: {duration:.2f} seconds")
                
                file_print(f"\nFinal Answer Generation: {timing_stats['final_answer']:.2f} seconds")
                
            # Print timing statistics
   
        except Exception as e:
            file_print(f"Error occurred: {e}")
            summary_print(f"{prob_num:^10} | {'Error':^10} | {'-':^10} | {problem['answer']:^10} | {'-':^11}")
            raise

        print(f"\nProcessing complete! Results saved to:")
        print(f"Detailed results: {detailed_file}")
        print(f"Summary results: {summary_file}")

    
    