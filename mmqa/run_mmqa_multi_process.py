import os
import json
import logging
from multiprocessing import Pool, Manager
from functools import partial

from tqdm import tqdm
from datasets import load_dataset

from parse_args import arg_parse
from run_mmqa import run_single_case, load_mmqa, get_output_file_path

def worker(example, args, lock):
    output_file_path = get_output_file_path(args)
    result = run_single_case((example, args))
    with lock:
        results = []
        if os.path.exists(output_file_path):
            with open(output_file_path) as f:
                results = json.load(f)
        results.append(result)
        with open(output_file_path, 'w') as f:
            json.dump(results, f)
    return result

if __name__ == "__main__":

    args = arg_parse()

    if args.logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info(f"Args: {args}")

    output_file_path = get_output_file_path(args)
    
    mmqa_dev = load_mmqa(args)

    if os.path.exists(output_file_path):
        finished_id = set()
        with open(output_file_path) as f:
            results = json.load(f)
        for result in results:
            finished_id.add(result['id'])
        mmqa_dev = [example for example in mmqa_dev if example['id'] not in finished_id]

    manager = Manager()
    lock = manager.Lock()

    output_file_path = os.path.join(args.output_path, 'predict.json')

    func = partial(worker, args=args, lock=lock)

    pool = Pool(args.process_num)
    generated_result = list(tqdm(pool.imap(func, mmqa_dev), total=len(mmqa_dev)))
    pool.close()
    pool.join()