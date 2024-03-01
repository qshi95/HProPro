# -*- coding: utf-8 -*-

# The built-in lib
from datetime import datetime
import json
import logging
from multiprocessing import Pool
import os
import sys
from tqdm import tqdm

# Lib in Project
from parse_args import arg_parse, read_config
from run_hybridqa import run_single_case


if __name__ == "__main__":

    if 'yaml' in sys.argv[1] or 'json' in sys.argv[1]:
        args = arg_parse(read_config(sys.argv[1]))
    else:
        args = arg_parse()

    if args.logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    print(f"Args: {args}")

    with open(args.data_path) as f:
        hybridqa_dev = json.load(f)

    hybridqa_dev_target = []
    if args.id != "":
        for x in hybridqa_dev:
            if x['question_id'] == args.id:
                hybridqa_dev_target.append(x)
    else:
        if args.end == -1: # Single case test
            end = args.start + 1
        elif args.end > len(hybridqa_dev):
            end = len(hybridqa_dev)
        else:
            end = args.end

        hybridqa_dev_target = hybridqa_dev[args.start:end]
    
    hybridqa_dev = hybridqa_dev_target

    now = datetime.now()
    dt_string = now.strftime("%m%d-%H%M")

    correct, wrong = 0, 0


    output_file_path = os.path.join(args.output_path, f'predictions.json')


    # Multi-threads process
    hybridqa_dev = list(zip(hybridqa_dev, [args for _ in range(len(hybridqa_dev))]))

    with Pool(args.num_processer) as p:
        # if args.retrieval:
        #     result = list(tqdm(p.imap(run_single_case_oracle, hybridqa_dev), total=len(hybridqa_dev)))
        # else:
        result = list(tqdm(p.imap(run_single_case, hybridqa_dev), total=len(hybridqa_dev)))

    print(result)

    with open(output_file_path, 'w')  as fout:
        json.dump(result, fout)
    logging.info(f"Dumped into {output_file_path}")

