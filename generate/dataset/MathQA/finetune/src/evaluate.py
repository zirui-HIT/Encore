import re
import sys
import json
import argparse

from re import RegexFlag
from copy import deepcopy
from typing import Any, Dict, List

sys.path.append(".")


def extract_structure_data(plain_text_content: str) -> List[Dict[str, Any]]:
    def sort_by_id(data):
        data.sort(key=lambda x: int(x.split('\t')[0][2:]))
        return data

    predict_outputs = sort_by_id(re.findall(
        "^D.+", plain_text_content, RegexFlag.MULTILINE))
    ground_outputs = sort_by_id(re.findall(
        "^T.+", plain_text_content, RegexFlag.MULTILINE))
    source_inputs = sort_by_id(re.findall(
        "^S.+", plain_text_content, RegexFlag.MULTILINE))

    def parse_answer(text: str) -> Dict[str, Any]:
        current_type = ''
        results: Dict[str, List[str]] = {
            "derivations": [],
            "variables": [],
            "answers": []
        }
        for word in text.split(' | '):
            word = word.strip()
            if word.startswith('<D>'):
                word = word[4:]
                if not word:
                    continue
                current_type = 'derivations'
            elif word.startswith('<V>'):
                word = word[4:]
                if not word:
                    continue
                current_type = 'variables'
            elif word.startswith('<A>'):
                word = word[4:]
                current_type = 'answers'
            results[current_type].append(word)

        if results["derivations"] and results['variables']:
            results['answers'] = deepcopy(results["derivations"])
            for i in range(len(results['variables']) - 1, -1, -1):
                for j in range(len(results['answers'])):
                    results['answers'][j] = results['answers'][j].replace(
                        f"x{i}", results['variables'][i])

        return {
            'derivations': results['derivations'],
            'variables': results['variables'],
            'answers': results['answers'],
            'origin_pred': text
        }

    idx = 0
    data: List[Dict[str, Any]] = []
    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        predict_id, predict_score, predict_clean = predict.split('\t')
        ground_id, ground_clean = ground.split('\t')
        source_id, source_clean = source.split('\t')

        real_pred_id = int(predict_id.split('-')[-1])
        while idx < real_pred_id:
            data.append({
                "id": f"D-{idx}",
                "pred": {},
                "gold": {},
                "source": ""
            })
            idx += 1

        try:
            pred_answer = parse_answer(predict_clean)
        except Exception as e:
            print(f"An error occurred in source: {idx}, {e}")
            pred_answer = {"origin_pred": predict_clean}
        try:
            gold_answer = parse_answer(ground_clean)
        except Exception as e:
            gold_answer = {"origin_gold": ground_clean}

        data.append({
            "id": predict_id,
            "pred": pred_answer,
            "gold": gold_answer,
            "source": source_clean
        })
        idx += 1

    return data


def process_file(generate_file_path: str, output_path: str) -> None:
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content)

    # evaluate execution accuracy
    match_count = 0
    for d in data:
        d['execution_correct'] = False
        if not ('answers' in d['gold']) or not ('answers' in d['pred']):
            continue
        if eval_execution_match(d['gold']['answers'], d['pred']['answers']):
            match_count += 1
            d['correct'] = True
        try:
            d['gold']['value'] = str(eval_calculate(d['gold']['answers']))
        except:
            d['gold']['value'] = None
        try:
            d['pred']['value'] = str(eval_calculate(d['pred']['answers']))
        except:
            d['pred']['value'] = None
    print(f"Execution EM: {match_count / len(data)}")

    # evaluate program match
    match_count = 0
    for d in data:
        d['program_correct'] = False
        if not ('answers' in d['gold']) or not ('answers' in d['pred']):
            continue
        if eval_program_match(d['gold']['answers'], d['pred']['answers']):
            match_count += 1
            d['program_correct'] = True
    print(f"Program EM: {match_count / len(data)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from generate.t5.dataset.MathQA.finetune.src.utils import eval_execution_match, eval_program_match, eval_calculate

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--origin_data_path", "-odp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--default_score", "-ds", type=float, default=None)
    args = parser.parse_args()

    process_file(args.data_path, args.output_path)
