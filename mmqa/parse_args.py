import json
import argparse
import yaml

def read_config(config_file_path):
      
    if '.json' in config_file_path:
        with open(config_file_path) as f:
            config = json.load(f)
    elif '.yaml' in config_file_path:
        with open(config_file_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise ValueError("The config file should be in the format of json or yaml.")
    return config

def arg_parse():
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument('--output_path', default='./result', type=str)
    parser.add_argument('--seed', default=4, type=int)

    # MODEL RUNNING
    parser.add_argument("--model", default='gpt4', type=str)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--reflection', action='store_true')
    parser.add_argument('--dry_run', action='store_true')


    # PROMPT
    parser.add_argument('--shot_num', default=4, type=int)
    parser.add_argument('--use_caption', action='store_true')
    parser.add_argument('--not_check', action='store_true')
    parser.add_argument('--use_cot', action='store_true')

    # RETRIEVE RERANK
    parser.add_argument('--use_oracle', action='store_true')
    parser.add_argument('--retrieve_top_k', default=10, type=int)
    parser.add_argument('--passage_rerank_top_k', default=3, type=int)
    parser.add_argument('--image_rerank_top_k', default=5, type=int)

    # SIMPLIFY
    parser.add_argument('--not_simplify', action='store_true')
    parser.add_argument('--with_row', action='store_true')

    # MULTI PROCESS
    parser.add_argument('--process_num', default=10, type=int)

    args = parser.parse_args()
    return args