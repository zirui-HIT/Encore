import re
import sys
import json
import argparse

from re import RegexFlag
from typing import Any, Dict, List

sys.path.append(".")


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
            if golden:
                raise e
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

        table = Tabular(origin_data[idx]['table']['table'])
        try:
            pred_answer = parse_answer(predict_clean, table)
        except Exception as e:
            print(f"An error occurred in source: {idx}, {e}")
            pred_answer = {"origin_pred": predict_clean}
        try:
            gold_answer = parse_answer(ground_clean, table, True)
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


def evaluate(data: List[Dict[str, Any]]) -> List[bool]:
    exec_count = 0
    prog_count = 0
    result: List[bool] = []
    for i, d in enumerate(data):
        if not ('answer' in gold_answers[i]):
            result.append(False)
            continue

        prog_flag = False
        exec_flag = False

        if 'base' in args.output_path and ' | '.join(d['pred']['answers']) == ' | '.join(d['gold']['answers']):
            prog_count += 1
            prog_flag = True
        elif ' | '.join(d['pred']['answers']) == ' | '.join(gold_answers[i]['answer']):
            prog_count += 1
            prog_flag = True
        else:
            pass

        if d['pred']['answer_value'] == gold_answers[i]['origin_answer_value']:
            exec_count += 1
            exec_flag = True
            result.append(True)
        elif isinstance(d['pred']['answer_value'], float) and isinstance(gold_answers[i]['origin_answer_value'], float) and abs(d['pred']['answer_value'] - gold_answers[i]['origin_answer_value']) < 1e-4:
            exec_count += 1
            exec_flag = True
            result.append(True)
        else:
            result.append(False)

        # if not exec_flag and prog_flag:
        #     print(d['source'])

    # print(f"exec acc : {exec_count / len(data)}")
    # print(f"prog acc : {prog_count / len(data)}")

    return result


def decompose(answers: List[str], table) -> List[str]:
    table: Tabular
    result: List[str] = []

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

        def update_value(value: str) -> str:
            if value.startswith('const_'):
                return value
            if value in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '100', '1000', '10000', '100000', '1000000', '10000000', '100000000']:
                value = f"const_{value}"
                return value
            if value.startswith('#'):
                return value
            if value.startswith('m'):
                return value

            value = value.replace('(', '').replace(')', '').strip()
            value = value.strip('$')
            for scale in ['thousand', 'million', 'billion']:
                value = value.strip(scale)
            if '%' in value:
                value = value.replace('%', '').strip()
                value = str(float(value) / 100)
            value = value.strip()

            float(value)
            return value

        if op in ['table_max', 'table_min', 'table_average', 'table_sum']:
            if not ('{ col' in arg1):
                arg1 = table.index(arg1)
            column_idx = int(arg1.split('{ col')[1].split(',')[0].strip())
            result.extend(
                [f"{op}(", table.get_item(column_idx, 0), 'none', ')'])
        elif op in ['add', 'subtract', 'multiply', 'divide', 'exp', 'greater']:
            value1 = update_value(value1)
            value2 = update_value(value2)
            result.extend([f"{op}(", value1, value2, ")"])
        else:
            raise Exception(f'illegal op : {op}')

    result = [str(int(x)) if isinstance(x, float) and x %
              1 == 0 else str(x) for x in result] + ['EOF']
    return result


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

    if not ('test' in output_path):
        correct_arr = evaluate(data)
        with open(output_path.replace('.json', '.cases.json'), 'w', encoding='utf-8') as f:
            for example, correct in zip(data, correct_arr):
                example['correct'] = correct
                example['pred']['answer_value'] = str(
                    example['pred']['answer_value'])
            json.dump([x for x in data if not x['correct']],
                      f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from generate.bart.dataset import Tabular
    from generate.t5.dataset.FinQA.finetune.src.utils import evaluate_result, calc_program

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--origin_data_path", "-odp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--default_score", "-ds", type=float, default=None)
    args = parser.parse_args()

    if not (args.origin_data_path is None):
        gold_answers: List[Dict[str, Any]] = []
        with open(args.origin_data_path, 'r', encoding='utf-8') as f:
            origin_data: List[Dict[str, Any]] = json.load(f)
            for d in origin_data:
                gold_answers.extend(d['questions'])

    process_file(args.data_path, args.output_path)
    if not ('test' in args.output_path):
        evaluate_result(
            args.output_path, f"/users10/dzrwang/HybridQA/dataset/FinQA/{args.origin_data_path.split('/')[-1]}")
