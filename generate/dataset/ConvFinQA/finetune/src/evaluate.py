import re
import sys
import json
import argparse

from re import RegexFlag
from typing import Any, Dict, List, Union

sys.path.append(".")


def calc_program(answers: List[str], table) -> Union[float, str]:
    table: Tabular
    result: List[Union[float, str]] = []

    for a in answers:
        op, other = a.split('(')
        other = other.strip(')').strip()

        parts: List[str] = other.split(',')
        if '{' in parts[0]:
            arg1 = f"{parts[0]},{parts[1]}"
        else:
            arg1 = parts[0]
        if '}' in parts[-1]:
            arg2 = f"{parts[-2]},{parts[-1]}"
        else:
            arg2 = parts[-1]

        op = op.strip()
        arg1 = arg1.strip()
        arg2 = arg2.strip()

        value1 = table.get(arg1)
        value2 = table.get(arg2)

        def update_value(value: str) -> float:
            if value.startswith('const_'):
                value = value[6:]
            if value.startswith('#'):
                return result[int(value.strip('#'))]
            elif value.startswith('m'):
                return -1
            else:
                return float(calc_derivation(value)[0])

        if op in ['table_max', 'table_min', 'table_average', 'table_sum']:
            if not ('{ col' in arg1):
                arg1 = table.index(arg1)
            column_idx = int(arg1.split('{ col')[1].split(',')[0].strip())
            if op == 'table_max':
                result.append(
                    max(table.table[column_idx][1:], key=(lambda x: calc_derivation(x)[0])))
            elif op == 'table_min':
                result.append(
                    min(table.table[column_idx][1:], key=(lambda x: calc_derivation(x)[0])))
            elif op == 'table_average':
                result.append(
                    sum([calc_derivation(x)[0] for x in table.table[column_idx][1:]]) / (table.row_number - 1))
            else:
                result.append(sum([calc_derivation(x)[0]
                              for x in table.table[column_idx][1:]]))
        elif op in ['add', 'subtract', 'multiply', 'divide', 'exp']:
            value1 = update_value(value1)
            value2 = update_value(value2)
            if op == 'add':
                result.append(value1 + value2)
            elif op == 'subtract':
                result.append(value1 - value2)
            elif op == 'multiply':
                result.append(value1 * value2)
            elif op == 'divide':
                result.append(value1 / value2)
            else:
                result.append(value1 ** value2)
        elif op in ['greater']:
            value1 = update_value(value1)
            value2 = update_value(value2)
            return "yes" if value1 > value2 else 'no'
        else:
            raise Exception(f'illegal op : {op}')

        if isinstance(result[-1], str):
            result[-1] = calc_derivation(result[-1])[0]

    return result[-1]


def extract_structure_data(plain_text_content: str, origin_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_by_id(data):
        data.sort(key=lambda x: int(x.split('\t')[0][2:]))
        return data

    predict_outputs = sort_by_id(re.findall(
        "^D.+", plain_text_content, RegexFlag.MULTILINE))
    ground_outputs = sort_by_id(re.findall(
        "^T.+", plain_text_content, RegexFlag.MULTILINE))
    source_inputs = sort_by_id(re.findall(
        "^S.+", plain_text_content, RegexFlag.MULTILINE))

    def parse_answer(text: str, table: Tabular, golden: bool = False) -> Dict[str, Any]:
        current_type = ''
        results: Dict[str, List[str]] = {
            "derivations": [],
            "variables": [],
            "answers": []
        }
        for word in text.split('|'):
            word = word.strip()
            if word.startswith('<D>'):
                word = word[4:]
                current_type = 'derivations'
            elif word.startswith('<V>'):
                word = word[4:]
                current_type = 'variables'
            elif word.startswith('<A>'):
                word = word[4:]
                current_type = 'answers'
            results[current_type].append(word)

        try:
            answer = calc_program(results['answers'], table)
        except Exception as e:
            # if golden:
            #     raise e
            answer = 0

        return {
            'answer_value': answer,
            'derivations': results['derivations'],
            'variables': results['variables'],
            'answers': results['answers'],
            'origin_pred': text
        }

    idx = 0
    data: List[Dict[str, Any]] = []
    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        predict_id, predict_score, predict_clean = predict.split('\t')
        ground_clean = '\t'.join(ground.split('\t')[1:])
        source_clean = '\t'.join(source.split('\t')[1:])

        real_pred_id = int(predict_id.split('-')[-1])
        while idx < real_pred_id:
            data.append({
                "id": f"D-{idx}",
                "pred": {},
                "gold": {},
                "source": ""
            })
            idx += 1

        table = Tabular(origin_data[idx]['table']['table'])
        try:
            pred_answer = parse_answer(predict_clean, table)
        except Exception as e:
            print(f"An error occurred in source: {idx}, {e}")
            pred_answer = {"origin_pred": predict_clean}
            # raise e
        try:
            gold_answer = parse_answer(ground_clean, table, True)
        except Exception as e:
            gold_answer = {"origin_gold": ground_clean}
            # raise e

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
        data = extract_structure_data(file_content, origin_data)

    result: List[Dict[str, Any]] = []
    for i, d in enumerate(data):
        try:
            result.append({
                "id": origin_data[i]['questions'][0]['uid'],
                "predicted": decompose(d['pred']['answers'], Tabular(origin_data[i]['table']['table']))
            })
        except Exception as e:
            print(e)
            result.append({
                "id": origin_data[i]['questions'][0]['uid'],
                "predicted": ""
            })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from generate.bart.dataset import Tabular
    from generate.utils import calc_derivation
    from generate.t5.dataset.ConvFinQA.finetune.src.utils import evaluate_result, decompose

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--origin_data_path", "-odp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()

    if not (args.origin_data_path is None):
        gold_answers: List[Dict[str, Any]] = []
        with open(args.origin_data_path, 'r', encoding='utf-8') as f:
            origin_data: List[Dict[str, Any]] = json.load(f)
            for d in origin_data:
                gold_answers.extend(d['questions'])

    process_file(args.data_path, args.output_path)
    if not ('test' in args.output_path):
        evaluate_result(args.output_path, args.origin_data_path,
                        args.output_path.replace(".json", ".cases.json"), "non-nest")
