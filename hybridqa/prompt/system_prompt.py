

SYSTEM_PROMPT = """Read the table and write python code to answer the given question. Note that sometimes answering questions requires extracting information in the hyperlinks. [HYPER] means these are hyperlinks in the cell. Assume we has defined the following functions to extract information from hyperlink text. You can use these functions directly as black box. You don't need to process the table. 

Attention: 
    1. The given table is in the pandas DataFrame format.
    2. The table has already contained the hyperlink in the cell. 
       a. If you want GET or SORT the content in the cell, you MUST use cell[0]. All data in cell[0] are string.
       b. If you want get the hyperlink string in the cell, you use cell[1]
    3. If you want to extract information from the hyperlink passages, you can use the function extract_info(cell[1], target information) defined above.
    4. If you want to compare two objects, you can use the function check(str1, str2, operation) defined above, don't use '==' or '<' or '>'.
"""

SYSTEM_PROMPT_COT = """Read the table and answer the given question step by step. Note that answering questions requires extracting information in the hyperlinks. [HYPER] means these are hyperlinks in the cell. Assume we has defined the following functions to extract information from hyperlink text. You can use these functions directly as black box. 
Attention: 
    1. If you want to extract information from the hyperlink passages, you can use the function extract_info(cell, target information) defined above.
    2. If you want to compare two objects, you can use the function check(str1, str2, operation) defined above, don't use '==' or '<' or '>'.
"""

SYSTEM_PROMPT_WO_CHECK = """Read the table and write python code to answer the given question. Note that sometimes answering questions requires extracting information in the hyperlinks. [HYPER] means these are hyperlinks in the cell. Assume we has defined the following functions to extract information from hyperlink text. You can use these functions directly as black box. You don't need to process the table. 

Attention: 
    1. The given table is in the pandas DataFrame format.
    2. The table has already contained the hyperlink in the cell. 
       a. If you want GET or SORT the content in the cell, you MUST use cell[0]. All data in cell[0] are string.
       b. If you want get the hyperlink string in the cell, you use cell[1]
    3. If you want to extract information from the hyperlink passages, you can use the function extract_info(cell[1], target information) defined above.
"""

SYSTEM_PROMPT_END2END = """Read the table and answer the given question.
"""


SYSTEM_PROMPT_v2 = """Read the table and write python code to answer the given question. Note that sometimes answering questions requires extracting information in the hyperlinks. [HYPER] means these are hyperlinks in the cell. Assume we has defined the following functions to extract information from hyperlink text. You can use these functions directly as black box. You don't need to process the table.
You should always use these functions to complete the code: 


def convert_time(time_str) -> str:
    # Convert any unregular date or time string to a datetime object.
def check(obj1, obj2, operation, model='gpt-3.5-turbo') -> bool:
    # Check if the obj1 and obj2 are describing the same thing.
    # Obj1 and obj2 could be anything.
    # operation should only be 'equal' or 'greater' or 'less'
def extract_info(cell: str, query: str) -> str:
    # extract `query` from hyperlink passages of the given `cell`.
    # `query` is a question asking what you want to know from the passage `cell`
Assume the table has already given in the pandas DataFrame format. You are required to complete the following function to solve this question. \n
def solve(table) -> str:
    # answer the question based on the given table. 
    return ans

Attention: 
    1. The given table is in the pandas DataFrame format.
    2. The table has already contained the hyperlink in the cell. 
       a. If you want GET or SORT the content in the cell, you MUST use cell[0]. All data in cell[0] are string.
       b. If you want get the hyperlink string in the cell, you use cell[1]
    3. If you want to extract information from the hyperlink passages, you can use the function extract_info(cell[1], target information) defined above.
    4. If you want to compare two objects, you can use the function check(str1, str2, operation) defined above, don't use '==' or '<' or '>'.
Now let's think step by step and write python code to solve it.
"""

SYSTEM_PROMPT_v1 = """Read the table and write python code to answer the given question. Note that sometimes answering questions requires extracting information in the hyperlinks. [HYPER] means these are hyperlinks in the cell. Assume we has defined the following functions to extract information from hyperlink text. You can use these functions directly as black box. You don't need to process the table.
You should always use these functions to complete the code: 


def convert_time(time_str) -> str:
    # Convert any unregular date or time string to a datetime object.
def check_same(obj1, obj2, model='gpt-3.5-turbo') -> bool:
    # Check if the obj1 and obj2 are describing the same thing.
def extract_info(cell: str, query: str) -> str:
    # extract `query` from hyperlink passages of the given `cell`.
    # `query` is a question asking what you want to know from the passage `cell`
Assume the table has already given in the pandas DataFrame format. You are required to complete the following function to solve this question. \n
def solve(table) -> str:
    # answer the question based on the given table. 
    return ans

[FEW_SHOT_PROMPT]

[TEST_PROMPT]

Attention: 
    1. The given table is in the pandas DataFrame format.
    2. The table has already contained the hyperlink in the cell. 
       a. If you want GET or SORT the content in the cell, you MUST use cell[0]. All data in cell[0] are string.
       b. If you want get the hyperlink string in the cell, you use cell[1]
    3. If you want to extract information from the hyperlink passages, you can use the function extract_info(cell[1], target information) defined above.
    4. If you want to compare two objects, you can use the function check_same(str1, str2) defined above, don't use '=='.
Now let's write python code to solve it. Please return the code directly without any explanation. Don't use the code block.
"""