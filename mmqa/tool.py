
import func_timeout
import re
import traceback


def safe_execute(code_string: str, para_dict: dict, keys: list=None) -> str:
    """Execute the input code string.

    Args:
        code_string (str): The str of the codes.
        para_dict (dict): The parameters and variables in the code.
        keys (list): The target return values. Defaults to None and return 'ans'.

    Returns:
        str: The string of executed result. 
            If executed properly, return the value of 'ans';
            If exception is thrown, return the text of traceback.
    """    
    def execute(x):
        try:
            exec(x, para_dict)
            if keys is None:
                return para_dict.get('ans', None)
            else:
                return [para_dict.get(k, None) for k in keys]
        except Exception as e:
            print(f"Traceback: {traceback.format_exc()}")
            return traceback.format_exc()
    try:
        ans_or_exception = func_timeout.func_timeout(600, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans_or_exception = "Function execute timeout."

    return ans_or_exception


def postprocess_code_new(result: str) -> str:
    
    delimiter_list = [' and ', ' or ']
    # delimiter_list = [' ']
    pattern = '|'.join(map(re.escape, delimiter_list))
    # print(pattern)
    lines = []
    
    for line in result.split('\n'):
        current_line = ""
        line = line.replace(" and ", " AND and ").replace(" or ", " OR or ")
        seg_list = re.split(pattern, line)
        for seg in seg_list:
            print(seg)
            if "==" in seg:
                print(seg_list)
                seg = seg.replace(" == ",".replace(")
                if seg.endswith(":"):
                    seg = seg[:-1]
                if " AND" in seg or " OR" in seg:
                    seg = seg.replace(" AND", ', "").startswith("/wiki") and')
                    seg = seg.replace(" OR", ', "").startswith("/wiki") or')
                else:
                    seg = seg + ', "").startswith("/wiki"):'
            current_line += seg + ' '
        lines.append(current_line)
    
    
    return '\n'.join(lines)


def postprocess_code(code: str) -> str:
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
        pattern = r"def solve(.*)    return (result|\"NOT_AVAILABLE\"|'NOT_AVAILABLE')"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            code_content = match.group(0)
            return code_content
        else:
            print(code)
            print("No match!")
            return None