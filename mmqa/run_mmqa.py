# -*- coding: utf-8 -*-

# The built-in lib
import os
import json
import logging
import random
from datetime import datetime
from typing import Dict, Any

# The third party lib
import inspect
from tqdm import tqdm
from datasets import load_dataset

# Lib in Project
from tool import *
import util
from util import *
from query_api import query_API
from retriever import Retriever
from prompt.system_prompt import SYSTEM_PROMPT_CAPTION, SYSTEM_PROMPT_CAPTION_WO_CHECK, SYSTEM_PROMPT_COT
from prompt.simplify_query import SIMPLIFY_QUERY_PASSAGE, SIMPLIFY_QUERY_IMAGE, SIMPLIFY_QUERY_CAPTION
from parse_args import arg_parse

args = {}

gpt3_token_consumption = 0
gpt4_token_consumption = 0

retriever = Retriever()

def get_output_file_path(args):
    output_file_path = './result/mmqa_result'
    output_file_path += '_o' if args.use_oracle else '_r'
    output_file_path += '_rf' if args.reflection else ''
    output_file_path += '_wos' if args.not_simplify else ''
    output_file_path += '_woc' if args.not_check else ''
    output_file_path += '_cot' if args.use_cot else ''
    output_file_path += f'_s{args.seed}' if args.seed > -1 else ''
    output_file_path += f'_{args.sample_num}' if args.sample_num > 0 else ''
    output_file_path += '.json'
    return output_file_path

def simplify(example, args):
    question = example['question']

    if args.use_oracle:
        retrieved_passages, retrieved_images = retriever.retrieve_oracle(example)
    else:
        retrieved_passages, retrieved_images = retriever.retrieve(example, 
                                                                  retrieve_top_k=args.retrieve_top_k, 
                                                                  passage_rerank_top_k=args.passage_rerank_top_k, 
                                                                  image_rerank_top_k=args.image_rerank_top_k)
    # passages
    if len(retrieved_passages) > 0:
        simplify_prompt = SIMPLIFY_QUERY_PASSAGE
        simplify_prompt = simplify_prompt.replace('[QUERY]', question)
        passage_prompts = []
        for idx, retrieved_passage in enumerate(retrieved_passages):
            passage_prompt = "Passage {0}:\nThis passage is about {1}. Its content is: {2}".format(idx+1, retrieved_passage['title'], retrieved_passage['text'])
            if args.with_row and retrieved_passage['cell'] != None:
                i, _ = retrieved_passage['cell']
                header = example['table']['header'][0]
                cell_info = example['table']['rows'][0][i]
                passage_prompt += "\nThe table row about this passage is:\n{0}\n{1}".format('|'.join(header), '|'.join(cell_info))
            passage_prompts.append(passage_prompt)
        simplify_prompt = simplify_prompt.replace('[KNOWLEDGE]', '\n'.join(passage_prompts))
        simplify_prompt = simplify_prompt.replace('[IMAGE]', '')

        simplify_question = query_API(simplify_prompt, model='gpt4')[0]
        if simplify_question is not None and simplify_question != "" and "NO_SIMPLIFY" not in simplify_question:
            question = simplify_question

    # images
    if len(retrieved_images) > 0:
        simplify_prompt = SIMPLIFY_QUERY_CAPTION if args.use_caption else SIMPLIFY_QUERY_IMAGE
        simplify_prompt = simplify_prompt.replace('[QUERY]', question)
        image_path = []
        image_prompts = []
        for idx, retrieved_image in enumerate(retrieved_images):
            if args.use_caption:
                image_prompt = "Image {0}:This image is about {1}. Here are the caption describe this image: {2}".format(idx+1, retrieved_image['title'], retrieved_image['caption'])
            else:
                image_path.append(retrieved_image['path'])
                image_prompt = "Image {0}:This image is about {1}.".format(idx+1, retrieved_image['title'])
        
            if args.with_row and retrieved_image['cell'] != None:
                i, _ = retrieved_image['cell']
                header = example['table']['header'][0]
                cell_info = example['table']['rows'][0][i]
                image_prompt += "\nThe table row about this image is:\n{0}\n{1}".format('|'.join(header), '|'.join(cell_info))
            image_prompts.append(image_prompt)
        simplify_prompt = simplify_prompt.replace('[KNOWLEDGE]', '')
        simplify_prompt = simplify_prompt.replace('[IMAGE]', '\n'.join(image_prompts))

        if args.use_caption:
            model = 'gpt4'
        else:
            model = 'gpt4v'
        simplify_question = query_API(simplify_prompt, image_path=image_path, model='gpt4')[0]
        if simplify_question is not None and simplify_question != ""and "NO_SIMPLIFY" not in simplify_question:
            question = simplify_question
    
    return question

def create_few_shot_code_prompt(shot_num: int, not_check: bool, use_cot: bool):
    """_summary_

    Args:
        shot_num (int): The num of the cases in the prompt in the few-shot settings.

    Returns:
        str: The string of the few-shot part.
    """

    prompt = ""
    few_shot_case_path = './few_shot/few_shot_case'
    if not_check:
        few_shot_case_path += '_wo_check'
    if use_cot:
        few_shot_case_path += '_cot'
    few_shot_case_path += '.json'

    with open(few_shot_case_path, 'r') as fin:
        shot_cases = json.load(fin)
    # shot_cases = random.sample(shot_cases, shot_num)
    shot_cases = list(shot_cases.values())[:shot_num]
    
    prompt += "--------------------\nFor Example:\n"

    for idx, case in enumerate(shot_cases):
        prompt += f'Case {idx+1}:\n'
        prompt += "[Table title]\n"
        prompt += case['table_title']
        prompt += "[Table]\n"
        prompt += case['table']
        prompt += "\n[Question]\n{}".format(case['question'])
        # if 'knowledge' in case.keys():
        #     # prompt += "[Retrieved]\n"
        #     prompt += "Besides, we have already retrieved some potential relevant knowledge to solve this question:\n"
        #     prompt += 'Knowledge regarding {} {}'.format(case['knowledge'].split('\t')[0], case['knowledge'].split('\t')[1])
        #     # prompt += case['knowledge']
        prompt += "\n[Code]\n"
        prompt += case['code']
        prompt += "\n\n"
    
    return prompt

def create_test_prompt(example: Dict[str, Any], args):
    """Generate the prompt for the test case.

    Args:
        example (Dict[str, Any]): an example from the hybridqa dataset (json file)

    Returns:
        str: The prompt of the test case
        pd.dataframe: The table in the pandas dataframe format.
    """

    prompt = "\nNow deal with this:\n"

    table_str, table_pd = linearize_table(example['table'])
    prompt += '[Table title]\n'
    prompt += example['table']['title'][0]
    prompt += '\n[Table]\n'
    prompt += table_str
    prompt += '\n'
    prompt += '[Question]\n'
    question = example['question']

    if not args.not_simplify:
        question = simplify(example, args)
    prompt += f'{question}\n'


    retrieved_passages, retrieved_images = retriever.retrieve_oracle(example)
    prompt += "\nBesides, I've found some related passages and images:\n"
    prompt += "Passages:\n"
    for passage in retrieved_passages:
        prompt += f"{passage['title']}: {passage['text']}" + "\n"
    
    prompt += "Images:\n"
    for image in retrieved_images:
        prompt += f"{image['title']}: {image['caption']}" + "\n"

    prompt += "[Code]\n"
    if args.use_cot:
        prompt += "Please list some equations to solve it, and mark the final answer as a variable named \"result\"\n"
    else:
        prompt += "First describe your solution to clarify your thinking, and write python code to solve this problem (only solve(table) function).\n"

    return prompt, table_pd

def execute(generated_code, table_pd):
    """Concat the util.py and the generated code , and execute the code generated. 

    Args:
        generated_code (str): The funciton "solve()" generated by GPT.

    Returns:
        str: Result get from the executed code.
    """

    # merge the code with util definition to enable execution with "exec" function
    ref_code = inspect.getsource(util)
    code = '\n'.join([ref_code, generated_code])

    logging.info("Dumping code...")
    with open("code_unprocess.py", 'w', encoding='utf-8') as f:
        f.write(code)

    logging.info("Running code...")
    ans = safe_execute(code+'\n\nans=solve(table)', {'table': table_pd})

    return ans

def run_single_case(example_and_args: tuple) -> dict:
    example, args = example_and_args
    full_prompt = SYSTEM_PROMPT_CAPTION 
    if args.not_check:
        full_prompt = SYSTEM_PROMPT_CAPTION_WO_CHECK
    if args.use_cot:
        full_prompt = SYSTEM_PROMPT_COT

    # Read the few-shot prompt
    few_shot_prompt = ""
    if args.shot_num > 0:   # In the few-shot setting
        few_shot_prompt = create_few_shot_code_prompt(args.shot_num, args.not_check, args.use_cot)
    
    full_prompt += '\n\n' + few_shot_prompt

    # Read the test prompt
    test_prompt, table_pd = create_test_prompt(example, args=args)

    full_prompt += '\n\n' + test_prompt

    logging.info(
        "===================\nTest prompt:\n-------------------\n"
        f"ID: {example['id']}\n{full_prompt}\n"
        "==================="
        )

    if args.dry_run:
        print(full_prompt)
        exit()

    """Run the model to generate the code."""
    response = query_API(full_prompt, model=args.model, temperature=args.temperature)[0]
    logging.info(
        "\n===================\nGenerated Content:\n-------------------\n"
        f"{response}\n===================\n"
    )

    response_code = parser_code_from_response(response)

    logging.info(
        "\n===================\nGenerated Code:\n-------------------\n"
        f"{response_code}\n===================\n"
    )

    ans = execute_generated_code(response_code, table_pd)

    traceback_record = ""
    if ans != "" and ans != None:
        if ans.startswith('Traceback'):
            traceback_record = ans
            ans = None
    
    logging.info(
        "=========================\n"
        f"answer: {ans}\n"
        f"Golden Answer: {example['answer_text']}\n"
        "=========================\n"
    )

    result = {
        'id': example['id'],
        'query': example['question'],
        'full_prompt': full_prompt,
        'code': response_code,
        'prediction': ans,
        'golden_answer': example['answer_text']
    }

    refined_code = refined_answer = ""

    if args.reflection and (
        ans in ["", "None", "NOT_AVAILABLE", "NOT_FOUND", "null", "Execute_Failed", "Not found"] or 
        ans == None
        ):
            logging.info(
                "\n=========================\n"
                "Reflection\n"
                "---------------------\n"
                f"trackback_record: {traceback_record}\n"
            )
            refined_code = refine(full_prompt, response_code, traceback_record)
            logging.info(
                "\n=========================\n"
                f"Refined Code:\n{refined_code}\n"
                "=========================\n"
            )
            refined_answer = ans if refined_code == response_code else execute_generated_code(refined_code, table_pd)

            logging.info(
                f"\nRefined answer: {refined_answer}\n"
                f"Golden Answer: {example['answer_text']}\n"
                "========================="
            )
            result['refined_code'] = refined_code
            result['refined_prediction'] = refined_answer
            result['traceback'] = traceback_record

    return result

def load_mmqa(args, split='validation'):
    if args.seed > -1:
        random.seed(args.seed)

    mmqa_dev = load_dataset(path='../data/mmqa.py', cache_dir='../data/mmqa_cache')[split]

    new_dataset_split_loaded = []
    for data_item in mmqa_dev:
        data_item['table']['page_title'] = data_item['table']['title']
        new_dataset_split_loaded.append(data_item)
    mmqa_dev = new_dataset_split_loaded
    if args.sample_num > 0:
        mmqa_dev = random.sample(mmqa_dev, args.sample_num)
    elif args.start > 0 or args.end > -1:
        mmqa_dev = mmqa_dev[args.start:args.end]

    return mmqa_dev

if __name__ == "__main__":

    args = arg_parse()

    if args.logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info(f"Args: {args}")

    mmqa_dev = load_mmqa(args)

    output_file_path = get_output_file_path(args)

    if os.path.exists(output_file_path):
        finished_id = set()
        with open(output_file_path) as f:
            results = json.load(f)
        for result in results:
            finished_id.add(result['id'])
        mmqa_dev = [example for example in mmqa_dev if example['id'] not in finished_id]
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")

    for idx, example in tqdm(enumerate(mmqa_dev), total=len(mmqa_dev)):
        result = run_single_case((example, args))
        if os.path.exists(output_file_path):
            with open(output_file_path) as f:
                results = json.load(f)
        results.append(result)
        with open(output_file_path, 'w') as f:
            json.dump(results, f)