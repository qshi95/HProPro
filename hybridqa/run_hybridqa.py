# -*- coding: utf-8 -*-

# The built-in lib
import os
import sys
import logging
from typing import Dict, Any

# The third party lib
from tqdm import tqdm

# Lib in Project
from utils import *
from query_api import query_API
from process_table import linearize_table
from parse_args import read_config, arg_parse
from prompt.system_prompt import SYSTEM_PROMPT
from prompt.rank_passage import PASSAGE_RANK_PROMPT
from prompt.simplify_query import SIMPLIFY_QUERY
from prompt.simplify_query import SIMPLIFY_QUERY



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
    prompt += "\nFirst describe your solution to clarify your thinking, and write python code to solve this problem like:\n```python\ndef solve(table) -> str:\n    xxx\n```\n"

    return prompt, table_pd



def rank_passages(query: str, passages: list) -> list:
    """LLM-Rank passages.

    Rank the most useful passage according to the query using LLM.

    The result is a list of int. If the regular expression does not match the result from the result, return an empty list.

    Args:
        query (str): the query
        passages (list): A list of the passage(text, not the links)

    Returns:
        list: The idx of the best passages.
    """

    prompt = PASSAGE_RANK_PROMPT
    prompt = prompt.replace("[QUERY]", query)
    retrieved_passage_str = ""
    for i in range(len(passages)):
        retrieved_passage_str += f"{i}. {passages[i]}\n"
    prompt = prompt.replace("[PASSAGES]", retrieved_passage_str)
    response = query_API(prompt, model='gpt4')
    pattern = r"So my answer (is|are) (.*?)\."
    match = re.search(pattern, response, re.DOTALL)

    if match:
        response = match.group(2)

    result = response.split(', ')
    result_int = []
    for i in result:
        if i.isdigit() and int(i) < len(passages):
            result_int.append(int(i))

    result = [passages[i] for i in result_int]

    return result


def retrieve_under_oracle(example: dict) -> list:
    """Get the retrieved knowledge with the trained retriever from S^3HQA

    Update 1.9:

    Args:
        example (dict): One single case.

    Returns:
        list: a list of retrieved knowledge text
    """

    question = example['question']

    row_pre = example['row_pre']

    
    # retrieved_links = [link for link, link_label in zip(links, link_labels) if link_label == 1]
    # Update 24.1.9: 他的retriever只有top-1是准的，所以直接用他的pre就行
    retrieved_links_idx = example['row_p_dict'][row_pre]
    all_links = example['links']

    retrieved_passages = [find_hyperlinks(all_links[link]) for link in retrieved_links_idx]

    retrieved_knowledge = rank_passages(question, retrieved_passages)

    return retrieved_knowledge


def create_test_prompt_oracle(example: dict, simplify=False) -> tuple:
    """Generate the prompt for the test case.

    Args:
        example (Dict[str, Any]): an example from the hybridqa dataset (json file)

    Returns:
        tuple[
            str: The prompt of the test case
            pd.DataFrame: The table in the pandas dataframe format.
        ]
    """    


    prompt = "\nNow deal with this:\n"

    table_id = example['table_id']
    # table, table_pd = linearize_input(find_table(args.resource_dir, table_id))
    table_str, table_pd = linearize_table(find_table(table_id))
    prompt += '[Table]\n'
    prompt += table_str
    prompt += '[Question]\n'
    question = example['question']
    
    """retrieve relevant knowledge
    Attention:
    Make retrieved_knowledge empty([]) is just for DEBUG. You should remove the DEBUG flag to retrieve relevant knowledge.
    """

    retrieved_knowledge = retrieve_under_oracle(example)
    
    if len(retrieved_knowledge) != 0 and simplify:
        simplify_prompt = SIMPLIFY_QUERY
        simplify_prompt = simplify_prompt.replace("[QUERY]", question)
        simplify_prompt = simplify_prompt.replace("[KNOWLEDGE]", '\n'.join(retrieved_knowledge))
        print(
            "\n-------------------\n"
            f"Simplify query: {simplify_prompt}\n"
            "-------------------\n"
        )
        question = query_API(simplify_prompt, model='gpt4')
        print(
            f"\nSimplified Question: {question}\n"
            "-------------------\n"
        )

    prompt += question

    if len(retrieved_knowledge) > 0:
        prompt += "\n[Passages]\nBesides, I have found some passages to help you generate the code:\n"
        prompt += '\n'.join(retrieved_knowledge)
    
    prompt += "\n[Code]\nFirst describe your solution to clarify your thinking, and write python code to solve this problem (only solve() function).\n"

    return prompt, table_pd



def run_single_case(example_and_args: tuple) -> dict:

    global args
    example, args = example_and_args

    # Load system prompt
    system_prompt = SYSTEM_PROMPT

    # init the prompt for GPT
    full_prompt = system_prompt

    # Read the few-shot prompt
    few_shot_prompt = ""
    if args.shot_num > 0:   # In the few-shot setting
        few_shot_prompt = create_few_shot_code_prompt(args.shot_num)

    full_prompt += "\n\n" + few_shot_prompt

    # Read the test prompt
    if args.oracle:
        test_prompt, table_pd = create_test_prompt_oracle(example, simplify=args.simplify)
    else:
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
        temperature=args.temperature
        )


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


    golden_answer = example['answer-text'] if 'answer-text' in example else None

    logging.info(
        "\n=========================\n"
        f"answer: {ans}\n"
        f"golden_answer = {golden_answer}\n"
        "=========================\n"
    )

    result = {
        'question_id': example['question_id'],
        'query': example['question'],
        'full_prompt': system_prompt + test_prompt,
        'code': response_code,
        'pred': ans,
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

            refined_answer = execute_generated_code(refined_code+"\n\nans=solve(table)", table_pd)
            if refined_answer is not None and str(refined_answer).startswith('Traceback'):
                refined_answer = None
            logging.info(
                f"\nRefined answer: {refined_answer}\n"
                f"Golden Answer: {golden_answer}\n"
                "========================="
            )
            result['refined_code'] = refined_code
            result['old_pred'] = result['pred']
            result['pred'] = refined_answer if refined_answer is not None else ""
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