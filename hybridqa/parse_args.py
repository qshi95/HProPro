
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


def arg_parse(config={}):
    parser = argparse.ArgumentParser()
    
    if len(config) > 0: # read from config file
        args = argparse.Namespace(**config)
        return args

    # DATA
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--id", default="", type=str)
    parser.add_argument("--data_path", default='data/hybridqa_dev.json', type=str)
    parser.add_argument("--output_path", type=str, default='outputs/')
    # parser.add_argument("--resource_dir", default='./data/WikiTables-WithLinks/', type=str)

    # MODEL RUNNING
    # parser.add_argument("--key", default='OPENAI_KEY', type=str)
    parser.add_argument("--task_name", default="", type=str)
    parser.add_argument("--dry_run", default=False, action='store_true')
    parser.add_argument("--model", default='gpt-3.5-turbo', type=str)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--dump_output", action='store_true')

    parser.add_argument("--reflection", action='store_true')
    parser.add_argument("--oracle", action='store_true')
    parser.add_argument("--simplify", action='store_true')
    
    parser.add_argument("--num_processer", default=10, type=int)
    parser.add_argument("--logging", action='store_true')

    # PROMPT
    # parser.add_argument("--full_table", action='store_true')
    parser.add_argument("--shot_num", default=0, type=int)
    parser.add_argument("--DEBUG", action='store_true')
    
    args = parser.parse_args()
    return args

