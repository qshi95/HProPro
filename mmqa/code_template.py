import os
import json
# from PIL import Image
import dateparser

from prompt.convert_datetime import CONVERT_DATETIME
from prompt.check_prompt import CHECK_PROMPT
from prompt.extract_info import EXTRACT_INFO_CAPTION, EXTRACT_IMAGE_INFO
from query_api import query_API
# from vision_language import vqa
from datetime import timedelta
# from vision_language import vqa

with open('./url_map.json') as f:
    url_map = json.load(f)

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
        op (str): The relation. (in ['==', '<', '>'])
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
    result = query_API(prompt, model=model)[0]
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

def extract_info_caption(cell, query) -> str:
    """Get the answer from the text from the hyperlink according to the given query.

    Args:
        cell (str): target cell in the table
        query (str): the target information we want the model to get
    """    
    # obtain the passages

    if cell == "" or cell == None:
        return "NOT_AVAILABLE"

    hyperlink = cell
    
    passage = "" if hyperlink not in url_map else url_map[hyperlink]['passage']
    #caption = "" if hyperlink not in url_map or len(url_map[hyperlink]['image']) == 0 else url_map[hyperlink]['image']['caption']
    caption = ""
    if hyperlink in url_map and len(url_map[hyperlink]['image']) > 0:
        if 'new_caption' in url_map[hyperlink]['image']:
            caption = url_map[hyperlink]['image']['new_caption']
        else:
            caption = vqa(Image.open(url_map[hyperlink]['image']['path']), "Please describe this image in as much detail as possible.")
            url_map[hyperlink]['image']['new_caption'] = caption
            with open('./url_map.json', 'w') as f:
                json.dump(url_map, f)

    # If you want to modify the text, go to 
    # prompt/extract_info.py

    prompt = EXTRACT_INFO_CAPTION
    
    #print(passage)
    prompt = prompt.replace("[PASSAGE]", passage)
    prompt = prompt.replace("[CAPTION]", f'Caption:\n{caption}')
    prompt = prompt.replace("[QUERY]", query)

    result = query_API(prompt, model='gpt4')

    return result[0]

def extract_image_info(cell, query:str) -> str:
    """Get the answer from the image from the hyperlink according to the given query.

    Args:
        cell (str): target cell in the table
        query (str): the target information we want the model to get
    """    
    if cell == "" or cell == None:
        return "NOT_AVAILABLE"

    hyperlink = cell
    
    passage = "" if hyperlink not in url_map else url_map[hyperlink]['passage']
    image_path = []
    if hyperlink in url_map and len(url_map[hyperlink]['image']) > 0:
        if 'path' in url_map[hyperlink]['image']:
            image_path.append(url_map[hyperlink]['image']['path'])

    # If you want to modify the text, go to 
    # prompt/extract_info.py

    prompt = EXTRACT_IMAGE_INFO
    
    #print(passage)
    prompt = prompt.replace("[PASSAGE]", passage)
    prompt = prompt.replace("[QUERY]", query)

    if len(image_path) > 0:
        result = query_API(prompt, image_path, model='gpt4v')
        #result = vqa(Image.open(image_path[0]), prompt)
    else:
        result = query_API(prompt, model='gpt4')

    return result[0]

extract_info = extract_image_info