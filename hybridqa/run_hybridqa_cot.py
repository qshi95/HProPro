# -*- coding: utf-8 -*-

# The built-in lib
import os
import sys
import logging
import re
from typing import Dict, Any

# The third party lib
from tqdm import tqdm

# Lib in Project
from utils import *
from query_api import query_API
from process_table import  linearize_table
from parse_args import read_config, arg_parse
from prompt.ans_post_process import ANS_POST_PROCESS_PROMPT
from prompt.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_v2, SYSTEM_PROMPT_COT
from prompt.simplify_query import SIMPLIFY_QUERY
from utils import parser_code_from_response
from code_template import extract_info




def create_test_prompt_cot(example: Dict[str, Any], simplify=False):
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
    prompt += '[Question]\n'
    # prompt += example['question']
    question = example['question']

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
        simplify_prompt = SIMPLIFY_QUERY
        simplify_prompt = simplify_prompt.replace("[QUERY]", question)
        simplify_prompt = simplify_prompt.replace("[KNOWLEDGE]", '\n'.join(retrieved_knowledge))
        question = query_API(simplify_prompt, model='gpt4')

    prompt += question
    prompt += "\n[Code]\n"
    prompt += "\n# Please list some equations to solve it, and mark the final answer as a variable named \"result\"\n"

    return prompt, table_pd



def refine(old_prompt, code_w_err, err_info, model='gpt-4'):
    """Call the api to refine the code generated from the api in the first round.

    Args:
        old_prompt (str): The origin input to generate the code
        code_w_err (str): The generated code when executed will get None
    """    
    
    system_prompt = "You are a helpful assistant."
    refine_query = "There is something wrong in your code, that I can't run it. Can you generate it again and fix it? Just foucus and rewrite the code. \n"
    if err_info != "":
        refine_query += "I got these information:\n"
        refine_query += err_info
    else:
        refine_query += "I just run your code, but the result is an empty string."
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': old_prompt},
        {'role': 'assistant', 'content': code_w_err},
        {'role': 'user', 'content': refine_query}
    ]
    reflected_response = query_API(messages, model=model)

    reflected_code = parser_code_from_response(reflected_response)

    return reflected_code

def run_single_case(example_and_args: tuple) -> dict:

    global args
    example, args = example_and_args
    full_prompt = SYSTEM_PROMPT_COT

    # Read the few-shot prompt
    few_shot_prompt = ""
    if args.shot_num > 0:   # In the few-shot setting
        few_shot_prompt = create_few_shot_code_prompt(args.shot_num, mode='cot')

    full_prompt += "\n\n" + few_shot_prompt

    test_prompt, table_pd = create_test_prompt_cot(example, simplify=args.simplify)

    if args.DEBUG:
        print("DEBUG")
        from code_unprocess import solve
        print(f"ID: {example['question_id']}")
        print(test_prompt)
        print(solve(table_pd))
        exit()

    full_prompt += "\n\n" + test_prompt

    if args.dry_run:
        print(full_prompt)
        print('=======================')
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
        temperature=args.temperature,
        n=1
        )

    logging.info(
        "\n===================\nGenerated Content:\n-------------------\n"
        f"{response}\n===================\n"
    )

    response_code = parser_code_from_response(response)
    print(response_code)

    ans = execute_generated_code(response_code, table_pd)

    traceback_record = ""
    if ans != "" and ans != None:
        if ans.startswith('Traceback'):
            traceback_record = ans
            ans = None


    golden_answer = example['answer-text'] if 'answer-text' in example else None

    logging.info(
        "\n=========================\n"
        f"answer: {ans}\n"
        f"golden_answer = {golden_answer}\n"
        "=========================\n"
    )

    result = {
        'id': example['question_id'],
        'query': example['question'],
        'full_prompt': SYSTEM_PROMPT + test_prompt,
        'code': response_code,
        'prediction': ans,
        'golden_answer': golden_answer
    }

    refined_code = refined_answer = ""

    if args.reflection and (
        ans in ["", "None", "NOT_AVAILABLE", "NOT_FOUND", "null", "Execute_Failed", ] or
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

            refined_answer = execute_generated_code(refined_code, table_pd)
            if refined_answer is not None and str(refined_answer).startswith('Traceback'):
                refined_answer = None
            logging.info(
                f"\nRefined answer: {refined_answer}\n"
                f"Golden Answer: {golden_answer}\n"
                "========================="
            )
            result['refined_code'] = refined_code
            result['refined_prediction'] = refined_answer
            result['traceback'] = traceback_record
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
            args.end = args.start + 1

        hybridqa_dev_target = hybridqa_dev[args.start:args.end]
    
    hybridqa_dev = hybridqa_dev_target

    now = datetime.now()
    datetime_string = now.strftime("%m%d-%H%M")

    correct, wrong = 0, 0
    em = 0
    generated_result = []

    origin_output_file_path = os.path.join(args.output_path, 'predictions.json')
    reflection_output_file_path = os.path.join(args.output_path, 'predictions_w_reflection.json')

    for id, example in tqdm(enumerate(hybridqa_dev), total=len(hybridqa_dev)):

        result = run_single_case((example, args))
        generated_result.append(result)
        if args.dump_output or args.reflection:
            with open(origin_output_file_path, 'w')  as fout:
                json.dump(generated_result, fout)
            logging.info(f"Dumped into {origin_output_file_path}\n---------------------")

    logging.info("All finished.")