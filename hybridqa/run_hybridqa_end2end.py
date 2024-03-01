# -*- coding: utf-8 -*-

# The built-in lib
import os
import re
import sys
import logging
from typing import Dict, Any

# The third party lib
from tqdm import tqdm

# Lib in Project
from utils import *
from query_api import query_API

from parse_args import read_config, arg_parse
from prompt.system_prompt import SYSTEM_PROMPT_END2END
from prompt.simplify_query import SIMPLIFY_QUERY
from process_table import linearize_table


def create_test_prompt(example: Dict[str, Any], simplify=False):
    """Generate the prompt for the test case.

    Args:
        example (Dict[str, Any]): an example from the hybridqa dataset (json file)

    Returns:
        str: The prompt of the test case
        pd.dataframe: The table in the pandas dataframe format.
    """    


    prompt = "\nNow deal with this:\n"

    table_id = example['table_id']
    # table, table_pd = linearize_input(find_table(args.resource_dir, table_id))
    table_str, table_pd = linearize_table(find_table(table_id))
    prompt += '[Table]\n'
    prompt += table_str

    """retrieve relevant knowledge
    Attention: 
    Make retrieved_knowledge empty([]) is just for DEBUG. You should remove the DEBUG flag to retrieve relevant knowledge.
    """

    from retriever import Question_Passage_Retriever
    qs_pas_retriever = Question_Passage_Retriever()

    retrieved_knowledge = qs_pas_retriever.retriever_hybridqa(example)     # need to modify when presenting a demo

    """
    Simplify the query.
    """
    if len(retrieved_knowledge) != 0 and simplify:
        prompt += "\nBesides, I've got some related knowledge for you:"
        prompt += '\n'.join(retrieved_knowledge)

    prompt += '\n[Question]\n'
    # prompt += example['question']
    question = example['question']
    prompt += question
    prompt += "\n[Code]\n"
    prompt += "\nYou can describe your solution to clarify your thinking, and give me you answer with `My answer is xxx.`\n"

    return prompt, table_pd


def run_single_case(example_and_args: tuple) -> dict:

    global args
    example, args = example_and_args

    # Load system prompt
    system_prompt = SYSTEM_PROMPT_END2END

    # init the prompt for GPT
    full_prompt = system_prompt

    # Read the few-shot prompt
    few_shot_prompt = ""
    if args.shot_num > 0:   # In the few-shot setting
        few_shot_prompt = create_few_shot_code_prompt(args.shot_num,  mode='end2end')

    full_prompt += "\n\n" + few_shot_prompt

    # Read the test prompt
    test_prompt, table_pd = create_test_prompt(example, simplify=args.simplify)

    if args.DEBUG:
        print("DEBUG")
        from code_unprocess import solve
        print(f"ID: {example['question_id']}")
        print(test_prompt)
        print(solve(table_pd))
        exit()

    full_prompt += "\n\n" + test_prompt

    if args.dry_run:
        print('=======================\nFull prompt:\n=======================')
        print(full_prompt)

        return

    logging.info(
        "\n===================\nTest prompt:\n-------------------\n"
        f"ID: {example['question_id']}\n{full_prompt}\n"
        "==================="
    )


    """Run the model to generate the code."""
    response = query_API(
        full_prompt,
        model=args.model,
        temperature=args.temperature
        )


    logging.info(
        "\n===================\nGenerated Content:\n-------------------\n"
        f"{response}\n===================\n"
    )

    pattern = r"My answer is (.+?)\."
    match = re.search(pattern, response, re.DOTALL)
    if match:
        answer = match.group(1)
    else:
        answer = "NOT_AVAILABLE"

    golden_answer = example['answer-text'] if 'answer-text' in example else None

    result = {
        'question_id': example['question_id'],
        'query': example['question'],
        'full_prompt': system_prompt + test_prompt,
        'pred': answer,
        'golden_answer': golden_answer
        }

    return result



if __name__ == "__main__":

    if 'yaml' in sys.argv[1] or 'json' in sys.argv[1]:
        args = arg_parse(read_config(sys.argv[1]))
    else:
        args = arg_parse()

    if args.logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    print(f"Args: {args}")

    with open(args.data_path) as f:
        hybridqa_dev = json.load(f)

    hybridqa_dev_target = []
    if args.id != "":
        for x in hybridqa_dev:
            if x['question_id'] == args.id:
                hybridqa_dev_target.append(x)
    else:
        if args.end == -1: # Single case test
            end = args.start + 1
        elif args.end > len(hybridqa_dev):
            end = len(hybridqa_dev)
        else:
            end = args.end

        hybridqa_dev_target = hybridqa_dev[args.start:end]
    
    hybridqa_dev = hybridqa_dev_target

    now = datetime.now()
    datetime_string = now.strftime("%m%d-%H%M")

    correct, wrong = 0, 0
    em = 0
    generated_result = []

    origin_output_file_path = os.path.join(args.output_path, 'predictions.json')

    for id, example in tqdm(enumerate(hybridqa_dev), total=len(hybridqa_dev)):

        result = run_single_case((example, args))
        generated_result.append(result)
        if args.dump_output or args.reflection:
            with open(origin_output_file_path, 'w')  as fout:
                json.dump(generated_result, fout)
            logging.info(f"Dumped into {origin_output_file_path}\n---------------------")

    logging.info("All finished.")