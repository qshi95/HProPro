import os
import json
from time import sleep
import dateparser
import re


from hybridqa.prompt.convert_datetime import CONVERT_DATETIME
from hybridqa.prompt.check_prompt import CHECK_PROMPT, CHECK_SAME_PROMPT
from hybridqa.prompt.extract_info import EXTRACT_INFO
from hybridqa.query_api import query_API




resource_path = './data/WikiTables-WithLinks/'
request_id_file_mapping = json.load(open('./request_id_file_mapping.json', 'r'))


def url2text(url):
    if url.startswith('https://en.wikipedia.org'):
        url = url.replace('https://en.wikipedia.org', '')
    return url.replace('/wiki/', '').replace('_', ' ')


def find_table(table_id):
    """Find the table according to the table id.

    Args:
        table_id (str): _description_

    Returns:
        dict: _description_
    """    
    if not table_id.endswith('.json'):
        table_id += '.json'

    table_file = os.path.join(resource_path, 'tables_tok/{}'.format(table_id))
    table = json.load(open(table_file, 'r'))

    return table


def find_hyperlinks(request_id: str):
    """Get hyperlink from the cell

    Args:
        request_id (str): ID of the file which the hyperlink is pointing to.

    Returns:
        dict: A json-dict object containing the content.
    """    
    request_file = request_id_file_mapping[request_id]
    hyperlinks = json.load(open(os.path.join(resource_path, 'request_tok/{}'.format(request_file)), 'r'))[request_id]
    return hyperlinks



def check_same(obj1, obj2, model='gpt-3.5-turbo'):
    """Query the model to compare the two object if they are the same.

    ----------Full Prompt---------------
    Please verify whether the semantics of the two strings are consistent. We don't need them to be exactly the same, as long as they contain semantic similarity. 
    Attention:
    0. If you think the two objects are always the same in any condition, return Ture; Else return False.
    1. "August" and "2023-08" can also be judged to be similar (you return true), because they are the same month; but if the precise date or month is DIFFERENT, you should return False.
    2. If it is other descriptive text(adj, adv), similar is ok. 
    3. "NOT_AVAILABLE" is different with any text, return False.

    Query : [QUERY]

    String 1 : [STRING1]

    String 2 : [STRING2]

    You can only return 'True' or 'False' directly without any explanation.
    ------------------------------------

    Args:
        obj1 (str): Object 1
        obj2 (str): Object 2
        model (str, optional): The uesd api. Defaults to 'gpt-3.5-turbo'.
    
    Returns:
        Bool: True if same else False.
    """    

    prompt = CHECK_SAME_PROMPT
    prompt = prompt.replace("[STRING1]", str(obj1))
    prompt = prompt.replace("[STRING2]", str(obj2))
    result = query_API(prompt, model=model)
    if result == 'True':
        return True
    elif result == 'False':
        return False


def check(obj1, obj2, op, model='gpt-3.5-turbo'):
    """Query the model to compare the two object

    ----------Full Prompt---------------
    Please verify whether the semantics of the two strings meet the given conditions.
    For example:
    1. Is ten equal to 9?  Return: "True"
    2. Is 21 Nov, 2030 < 09-31-2021 ?  Return: "False"
    3. Is Beijing = The capital of China? Return: "True"
    4. Is 
    Attention:
    0. If you think the two objects are always the same in any condition, return Ture; Else return False.
    1. "August" and "2023-08" can also be judged to be similar (you return true), because they are the same month; but if the precise date or month is DIFFERENT, you should return False.
    2. If it is other descriptive text(adj, adv), similar is ok. 
    3. "NOT_AVAILABLE" is different with any text, return False.

    Query :  Is [STRING1] [REL] [STRING2] ?

    You can only return 'True' or 'False' directly without any explanation.
    ------------------------------------

    Args:
        obj1 (str): Object 1
        obj2 (str): Object 2
        op (str): The relation. (in ['=', '<', '>'])
        model (str, optional): The used model. Defaults to 'gpt-3.5-turbo'.
    
    Returns:
        Bool: True if the relation is tenable else False.
    """
    prompt = ""
    if op not in ['==', '>', '<']:
        return False
    prompt = CHECK_PROMPT
    prompt = prompt.replace("[STRING1]", str(obj1))
    prompt = prompt.replace("[STRING2]", str(obj2))
    prompt = prompt.replace("[REL]", str(op))
    result = query_API(prompt, model=model)
    if result == 'True':
        return True
    else:
        return False


def convert_time(time_str):
    """Use the model to convert a unregular time string to a datetime object.

    Args:
        time_str (str): The string containing a time.

    Returns:
        str: A datetime
    """    
    prompt = CONVERT_DATETIME
    prompt = prompt.replace("[TIME]", time_str)
    result = query_API(prompt)
    print(f"result: {result}")
    result = dateparser.parse(result)
    return result

def extract_info(cell, query):
    """Get the answer from the text from the hyperlink according to the given query.

    Args:
        cell (str): target cell in the table
        query (str): the target information we want the model to get
    """    
    # obtain the passages

    # pattern = r'%s(.*?)%s' % (re.escape('['), re.escape(']'))
    # match = re.search(pattern, cell)
    # hyperlinks = match.group(1).replace("\"", "").replace("\'", "").strip().split(',')

    # TODO:
    # 处理cot中对于extract_info的调用方法
    global table
    if cell == "" or cell == None:
        return "NOT_AVAILABLE"
    
    if not cell.startswith('/wiki/'):
        cell_link = []
        for index, row in table.iterrows():
            for column in table.columns:
                cell_value = row[column]
                if cell_value[0] == cell:
                    cell_link.append(cell_value[1])

        cell = '###'.join(cell_link)

    # hyperlinks = cell.strip().split(',')
    # hyperlinks = [item.strip() for item in hyperlinks]

    # cell_content = cell[0]
    # hyperlink_str = cell[1]

    cell_content = ""
    hyperlink_str = cell
    hyperlinks = hyperlink_str.split("###")
    for item in hyperlinks:
        if not item.startswith('/wiki/'):
            raise Exception('The first input value is not a hyperlink or a value in the table, you should check the parameter `cell`')

    passages = '\n'.join([find_hyperlinks(item) for item in hyperlinks])

    # cell_title = ' and '.join([url2text(t) for t in hyperlinks])
    # passages = "Extract info from the passage regarding " + cell_title + ": " + passages

    # If you want to modify the text, go to 
    # prompt/extract_info.py

    prompt = EXTRACT_INFO
    
    prompt = prompt.replace("[CELL_CONTENT]", cell_content)
    prompt = prompt.replace("[PASSAGES]", passages)
    prompt = prompt.replace("[QUERY]", query)

    # result = query_API(prompt)
    
    # with open("prompt/ans_post_process.txt", 'r') as f:
    #     prompt = f.read()
    
    # prompt = prompt.replace("[QUERY]", query)
    # prompt = prompt.replace("[CONTEXT]", result)

    result = query_API(prompt, model='gpt4')

    pattern = r"So my answer is (.*?)\."
    match = re.search(pattern, result, re.DOTALL)
    if match:
        result = match.group(1)
    else:
        result = "NOT_AVAILABLE"

    return result

