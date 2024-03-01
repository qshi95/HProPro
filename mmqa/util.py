import pandas as pd
import re
import inspect
import code_template as code_template
from code_template import *
import dateparser
from tool import safe_execute
from datetime import date, datetime, timedelta
from prompt.ans_post_process import ANS_POST_PROCESS_PROMPT

def process_cell_content(cell_content):
    """Process the content of the cell, mainly fouced on :
    1. "," in the number: 12,345
    2. Type of the content:
        a. int
        b. float
        c. datetime(NOT DONE YET)
    Args:
        cell_content (str): The str of the content of the origin cell.
    """    
    if ',' in cell_content and cell_content.replace(',', '').strip().isdigit(): 
        # Deal with : 
        #  Int with 10,000
        cell_content = cell_content.replace(',', '').strip()
    if '.' in cell_content and cell_content.replace('.', '').strip().isdigit(): 
        # Deal with:
        #   Float: 1.2345
        #   Datetime: 21.1.2000/2000.1.21
        if len(cell_content.split('.')) > 2:
            if len(cell_content.split('.')[0]) == 4:
                pattern = r'%Y.%m.%d'
                cell_content = datetime.strptime(cell_content, pattern)
            elif len(cell_content.split('.')[-1]) == 4:
                pattern = r'%d.%m.%Y'
                cell_content = datetime.strptime(cell_content, pattern)
            else:
                cell_content = cell_content
        else:
            cell_content = float(cell_content)
    elif cell_content.isdigit():
        cell_content = int(cell_content)
    
    else:
        # Deal with:
        #   Datetime: 2019-01-01
        # Split the cell_content by '-', '/' and ':', if all the parts are digit, then try to convert them to datetime
        is_digital = False
        if '-' in cell_content or '/' in cell_content or ':' in cell_content:
            #print(cell_content)
            is_digital = True
            cell_content_split = re.split('-|/|:|\.', cell_content)
            for split_part in cell_content_split:
                if not split_part.isdigit():
                    is_digital = False
                    break
        if is_digital:
            #print(cell_content)
            try:
                pattern = r"\d+:\d+\.\d+"
                if re.match(pattern, cell_content):
                    cell_content = "0:" + cell_content
                cell_content = dateparser.parse(cell_content)
            except Exception:
                return cell_content
    return cell_content

def linearize_table(table):
    """Linearize the table to a string, and build a pandas DataFrame for the table.

    Args:
        table (dict): a dict object, loaded from dataset file.

    Returns:
        str, pd.dataframe: The linearized string of the table, and the pandas DataFrame of the table.
    """    
    table_list = []
    table_linearized_str = ''
    
    # Get headers, rows, links
    headers = table['header'][0]
    rows = table['rows'][0]
    rows_with_links = table['rows_with_links'][0]
    # Zip the headers and hyperlinks of the headers. Note there are no hyperlinks of headers in mmqa
    headers = [str(process_cell_content(item)) for item in headers]

    # Add the header to the table_list
    table_list.append([str(item).strip() for item in headers])
    # Linearize the headers
    headers = 'col : ' + ' | '.join(headers)

    # Deal with the data in table
    table_row_str_list = []
    for i, (row, rwl) in enumerate(zip(rows, rows_with_links)):
        content = [process_cell_content(cell) for cell in row]
        rwl = [item[-1] for item in rwl]
        hyper = ['' if len(cell)==0 else str(cell) for cell in rwl]

         # item2[2:-2] is to delete the '[]' and '""'
        table_row = [[item1, item2[2:-2]] for item1, item2 in zip(content, hyper)]
        table_list.append(table_row)

        table_row_str = 'row {} : '.format(i+1) + ' | '.join([str(cell[0]) + ' ["' + str(cell[1]) + '"]' for cell in table_row])

        table_row_str_list.append(table_row_str)

    table_linearized_str += headers + '\n'
    table_linearized_str += '\n'.join(table_row_str_list)

    table_pd = pd.DataFrame(table_list[1:], columns=table_list[0])

    return table_linearized_str, table_pd

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
    if code is None:
        return "def solve(table):\n    return None"
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
            return "def solve(table):\n    return None"

def execute_generated_code(generated_code, table_pd):
    """Concat the util.py and the generated code , and execute the code generated. 

    Args:
        generated_code (str): The funciton "solve()" generated by GPT.

    Returns:
        str: Result get from the executed code.
    """

    # merge the code with util definition to enable execution with "exec" function
    ref_code = inspect.getsource(code_template)
    code = ref_code + '\n' + generated_code + '\n\n' + "ans=solve(table)"

    with open("code_unprocess.py", 'w', encoding='utf-8') as f:
        f.write(code)

    ans = safe_execute(code, {'table': table_pd})

    if isinstance(ans, datetime):
        ans = ans.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(ans, date):
        ans = ans.strftime('%Y-%m-%d')
    elif isinstance(ans, timedelta):
        ans = ans.total_seconds()

    ans = str(ans)
    return ans

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
    reflected_response = query_API(messages, model=model)[0]

    reflected_code = parser_code_from_response(reflected_response)

    return reflected_code

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