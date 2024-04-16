import json
from parse_args import arg_parse
from evaluator import evaluate_predictions

def eval_mmqa(results):
    pred_dict = {eid: [str(result['prediction'] if 'refined_prediction' not in result else result['refined_prediction'])] for eid, result in enumerate(results)}
    gold_dict = {eid: result['golden_answer'].split('|') for eid, result in enumerate(results)}
    eval_score, instance_eval_results = evaluate_predictions(pred_dict, gold_dict)
    for key in eval_score:
        eval_score[key] *= 100
    return eval_score, instance_eval_results

if __name__ == '__main__':
    args = arg_parse()
    result_path = args.output
    with open(result_path) as f:
        results = json.load(f)
    eval_score, instance_eval_results = eval_mmqa(results)
    print(eval_score)