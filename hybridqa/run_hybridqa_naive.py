# -*- coding: utf-8 -*-

# The built-in lib
from typing import Dict, Any

# The third party lib


# Lib in Project
from utils import *
from code_template import find_table
from process_table import linearize_table
from query_api import query_API


from prompt.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_v2
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

