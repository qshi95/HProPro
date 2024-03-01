
import func_timeout
import re
import traceback
import json
import inspect
import code_template as code_template
from code_template import *
from datetime import date, datetime, timedelta
from prompt.ans_post_process import ANS_POST_PROCESS_PROMPT
from query_api import query_API


def execute_generated_code(generated_code, table_pd):
    """Concat the util.py and the generated code , and execute the code generated. 

    Args:
        generated_code (str): The code generated by GPT.

    Returns:
        str: Result get from the executed code.
    """

    # merge the code with util definition to enable execution with "exec" function
    ref_code = inspect.getsource(code_template)
    code = ref_code + '\n' + generated_code + '\n\n'
    if "def solve" in generated_code:
        code += "result=solve(table)"

    with open("code_unprocess.py", 'w') as f:
        f.write(code)

    print(table_pd)
    ans = safe_execute(code, {'table': table_pd})

    if isinstance(ans, datetime):
        ans = ans.shrftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(ans, date):
        ans = ans.shrftime('%Y-%m-%d')
    elif isinstance(ans, timedelta):
        ans = ans.total_seconds()

    ans = str(ans)
    return ans


def safe_execute(code_string: str, para_dict: dict, keys: list=None) -> str:
    """Execute the input code string.

    Args:
        code_string (str): The str of the codes.
        para_dict (dict): The parameters and variables in the code.
        keys (list): The target return values. Defaults to None and return 'ans'.

    Returns:
        str: The string of executed result. 
            If executed properly, return the value of 'result';
            If exception is thrown, return the text of traceback.
    """
    def execute(x):
        try:
            exec(x, para_dict)
            if keys is None:
                return para_dict.get('result', None)
            else:
                return [para_dict.get(k, None) for k in keys]
        except Exception as e:
            print(f"Traceback: {traceback.format_exc()}")
            return traceback.format_exc()
    try:
        ans_or_exception = func_timeout.func_timeout(300, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans_or_exception = "Execute_Failed"

    return ans_or_exception



def parser_code_from_response(code: str) -> str:
    """Extract python code from the input mass.

    Extract the python code in the format of
    
    ```python
        a = 0
    ```
    
, or
    
    ```
        a = 0
    ```
    
, or
    
    def solve
        xxx
        return (result|\"NOT_AVAILABLE\"|'NOT_AVAILABLE')
    

    Args:
        result (str): The content generated from the api. There might be some explanation and analysis here. 

    Returns:
        str: The string of python code.
    """    
    
    if code.startswith('def solve('):
        return code
    pattern = r"\```python\n(.+?)\n```" if "```python" in code else r"\```\n(.+?)\n```"
    match = re.search(pattern, code, re.DOTALL)
    if match:
        code_content = match.group(1)
        return code_content
    else:
        pattern = r"def solve(.+?)    return (result|\"NOT_AVAILABLE\"|'NOT_AVAILABLE')"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            code_content = match.group(0)
            return code_content
        else:
            print(code)
            print("No match!")
            return "def solve():\n    return None"


def create_few_shot_code_prompt(shot_num, mode='pot'):
    """_summary_

    Args:
        shot_num (int): The num of the cases in the prompt in the few-shot settings.

    Returns:
        str: The string of the few-shot part.
    """

    prompt = ""

    if mode == "pot":
        case_file_link = 'hybridqa/few_shot_case/code_few_shot_cases_w_check.json'
    elif mode == "cot":
        case_file_link = "hybridqa/few_shot_case/code_few_shot_cases_w_check_cot.json"
    elif mode == "end2end":
        case_file_link = "hybridqa/few_shot_case/code_few_shot_cases_end2end.json"
    else:
        print(f"Wrong mode : {mode}")
        return

    with open(case_file_link, 'r') as fin:
        shot_cases = json.load(fin)
    # shot_cases = random.sample(shot_cases, shot_num)
    shot_cases = shot_cases[:shot_num]
    
    prompt += "--------------------\nFor Example:\n"

    for case in shot_cases:
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

def ans_post_process(query, ans_raw):
    """Post process the answer generated by the model.

    Args:
        query (str): The question of the test case.
        ans_raw (str): The answer executed by the generated code.

    Returns:
        str: The answer post processed by the gpt-3.5-turbo model.
            Note: `ans_processed` is not always different with the `ans_raw`.
    """

    ans_process_prompt = ANS_POST_PROCESS_PROMPT
    ans_process_prompt = ans_process_prompt.replace('[QUERY]', str(query))
    ans_process_prompt = ans_process_prompt.replace('[ANSWER]', str(ans_raw))
    ans_processed = query_API(ans_process_prompt, model='gpt4')
    return ans_processed


def refine(old_prompt, code_w_err, err_info, model='gpt-4'):
    """Call the api to refine the code generated from the api in the first round.

    Args:
        old_prompt (str): The origin input to generate the code
        code_w_err (str): The generated code when executed will get None
    """    
    
    system_prompt = "You are a helpful assistant."
    refine_query = "There is something wrong in your code, that I can't run it. Can you generate it again and fix it? Just foucus and rewrite the solve() function.\n"
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


def major_voting(ans_list: list) -> str:
    """voting for the answer appearing the highest frequency

    Args:
        ans_list (list): the answers from different codes

    Returns:
        str: the answers and the frequency(sorted)
    """    
    result_set = {}
    for ans in ans_list:
        if ans is None:
            continue
        if ans not in result_set:
            result_set[ans] = 0
        result_set[ans] += 1

    sorted_dict = dict(sorted(result_set.items(), key=lambda item: item[1], reverse=True))
    print(sorted_dict)
    final_answer = list(sorted_dict.keys())[0]
    return final_answer


