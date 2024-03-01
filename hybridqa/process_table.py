# -*- coding: utf-8 -*-
import re

import dateparser
import pandas as pd


def process_cell_content(cell_content, strlize=False):
    """Process the content of the cell, mainly fouced on :
    1. "," in the number: 12,345
    2. Type of the content:
        a. int
        b. float
        c. datatime(NOT DONE YET)
    Args:
        cell_content (str): The str of the content of the origin cell.
    """
    # print(cell_content)
    if ',' in cell_content and cell_content.replace(',', '').strip().isdigit(): 
        # Deal with : 
        #  10,000 -> 10000
        cell_content = cell_content.replace(',', '').strip()

    if '.' in cell_content and cell_content.replace('.', '').strip().isdigit(): 
        # Deal with:
        #   Float: 1.2345
        # cell_content = float(cell_content)
        return cell_content if strlize else float(cell_content)
    
    if cell_content.isdigit():
        # cell_content = int(cell_content)
        return cell_content if strlize else int(cell_content)
    
    else:
        # Deal with:
        #   Datetime: 2019-01-01
        # Split the cell_content by '-', '/' and ':', if all the parts are digit, then try to convert them to datetime
        is_digital = False
        if '-' in cell_content or '/' in cell_content or ':' in cell_content:
            is_digital = True
            cell_content_split = re.split('-|/|:|\.', cell_content)
            for split_part in cell_content_split:
                if not split_part.isdigit():
                    is_digital = False
                    break
        if is_digital:
            try:
                pattern = r"^[0-5]?\d:[0-5]?\d[.]?[\d]*$"
                if re.match(pattern, cell_content):
                    cell_content = "0:" + cell_content
                cell_content = dateparser.parse(cell_content)
            except Exception:
                return cell_content
    return str(cell_content) if strlize else cell_content

def linearize_table(table, full_table=True):
    """Linearize the table to a string, and build a pandas DataFrame for the table.

    Args:
        table (list): a list object, loaded from the json file of the table.

    Returns:
        str, pd.dataframe: The linearized string of the table, and the pandas DataFrame of the table.
    """    
    table_list = []
    table_linearized_str = ''

    # Deal with the header of the table
    # Get the string of the headers
    headers = [cell[0] for cell in table['header']]
    # Get the hyperlinks of the headers
    # hyper = ['[]' if len(cell[1])==0 else '[HYPER]' for cell in table['header']]
    # Zip the headers and hyperlinks of the headers
    # headers = [str(item1)+item2 for item1, item2 in zip(headers, hyper)]
    # Add the header to the table_list
    table_list.append([item.strip() for item in headers])
    # Linearize the headers
    headers = 'col : ' + ' | '.join(headers)

    # Deal with the data in table
    table_row_str_list = []
    for i, row in enumerate(table['data']):
        # Process the type and ',' in the content of the cell
        # content = [cell[0] for cell in row]
        content = [process_cell_content(cell[0], strlize=True) for cell in row]
        # content = [str(cell[0]) for cell in row]
        hyper = ['' if len(cell[1])==0 else '###'.join(cell[1]) for cell in row]
    
        # item2[2:-2] is to delete the '[]' and '""'
        table_row = [[item1, item2] for item1, item2 in zip(content, hyper)]
        table_list.append(table_row)
    
        # table_row_str = 'row {} : '.format(i+1) + ' | '.join([str(cell[0]) + ' ["' + str(cell[1]) + '"]' for cell in table_row])
        hyper_str = ['[]' if len(cell[1])==0 else '[HYPER]' for cell in row]
        table_row_str = 'row {} : '.format(i+1) + ' | '.join([str(cell[0]) + ' ' + hyper for cell, hyper in zip(table_row, hyper_str)])
        # table_row_str = 'row {} : '.format(i+1) + ' | '.join([str(cell[0])  + ' [""]' for cell in table_row])
        # table_row_str += ' | '.join()

        table_row_str_list.append(table_row_str)
        if not full_table and i >= 2:
            break

    table_linearized_str += headers + '\n'
    table_linearized_str += '\n'.join(table_row_str_list) + '\n'

    table_pd = pd.DataFrame(table_list[1:], columns=table_list[0])

    return table_linearized_str, table_pd
