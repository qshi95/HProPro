import argparse

def arg_parse():
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=0, type=int)
    parser.add_argument('--sample_num', default=0, type=int)
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--replicate', action='store_true')

    # MODEL RUNNING
    parser.add_argument('--model', default='gpt4', type=str)
    parser.add_argument('--temperature', default=0, type=float)
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--reflection', action='store_true')
    parser.add_argument('--ans_postprocess', action='store_true')

    # PROMPT
    parser.add_argument('--shot_num', default=4, type=int)
    parser.add_argument('--use_caption', action='store_true')
    parser.add_argument('--use_cot', action='store_true')
    parser.add_argument('--not_check', action='store_true')

    # RETRIEVE RERANK
    parser.add_argument('--use_oracle', action='store_true')
    parser.add_argument('--retrieve_top_k', default=10, type=int)
    parser.add_argument('--passage_rerank_top_k', default=3, type=int)
    parser.add_argument('--image_rerank_top_k', default=5, type=int)

    # SIMPLIFY
    parser.add_argument('--not_simplify', action='store_true')

    # MULTI PROCESS
    parser.add_argument('--process_num', default=10, type=int)

    # RESULT
    parser.add_argument('-o', '--output', default='./mmqa_result.json', type=str)

    args = parser.parse_args()
    return args