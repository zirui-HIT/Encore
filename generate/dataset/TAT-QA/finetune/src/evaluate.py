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

    def parse_answer(text: str,
                     tabular: Tabular) -> Dict[str, Any]:
        answers: List[str] = []
        answer_type = ""
        scale = ""
        derivation = ""
        current_type = 'None'
        variables: List[str] = []
        for word in text.split('|'):
            word = word.strip()
            if word.startswith('<C>'):
                answer_type = word.strip('<C>').strip()
                current_type = 'C'
            elif word.startswith('<D>'):
                derivation = word.strip('<D>').strip()
                current_type = 'D'
            elif word.startswith('<V>'):
                variables.append(word.strip('<V>').strip())
                current_type = 'V'
            elif word.startswith('<A>'):
                answers.append(word.strip('<A>').strip())
                current_type = 'A'
            elif word.startswith('<S>'):
                scale = word.strip('<S>').strip()
                current_type = 'S'
            elif current_type == 'A':
                answers.append(word.strip())
            elif current_type == 'V':
                variables.append(word.strip())

        if not answers:
            answers.append('0')

        if answer_type in ['span', 'count', 'multi-span']:
            if len(answers) == 1 and answers[0].endswith(scale):
                scale = ""
            if answer_type == 'count' and not answers[0].isdigit():
                if len(answers) == 1:
                    answer_type = 'span'
                else:
                    answer_type = 'multi-span'
            return {
                'type': answer_type,
                'answer': answers,
                'scale': scale,
                'origin_pred': text
            }
        elif answer_type in ['arithmetic']:
            try:
                for v in variables:
                    answers[0] = answers[0].replace(v, tabular.get(v))
                derivation = answers[0]

                answer, all_percent = calc_derivation(derivation, scale)
                if scale == 'percent' and ('/' in derivation or all_percent):
                    answer *= 100
            except Exception as e:
                answer = answers[0]
            return {
                'type': answer_type,
                'answer': str(answer),
                'scale': scale,
                "derivation": derivation,
                'origin_pred': text
            }
        else:
            raise Exception(f'illegal answer type : {answer_type}')

    def index_table(data: List[Dict[str, Any]], idx: int) -> Tabular:
        for d in data:
            if len(d['questions']) > idx:
                return Tabular(d['table']['table'])
            idx -= len(d['questions'])

    idx = 0
    data: List[Dict[str, Any]] = []
    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        try:
            predict_id, predict_score, predict_clean = predict.split('\t')
            ground_id, ground_clean = ground.split('\t')
            source_id, source_clean = source.split('\t')
        except:
            data.append({
                "id": f"D-{idx}",
                "pred": {},
                "gold": {},
                "source": ""
            })
            idx += 1
            continue

        real_pred_id = int(predict_id.split('-')[-1])
        while idx < real_pred_id:
            data.append({
                "id": f"D-{idx}",
                "pred": {},
                "gold": {},
                "source": ""
            })
            idx += 1

        table = index_table(origin_data, idx)
        try:
            pred_answer = parse_answer(predict_clean, table)
        except Exception as e:
            print(f"An error occurred in source: {idx}, {e}")
            pred_answer = {"origin_pred": predict_clean}
        try:
            gold_answer = parse_answer(ground_clean, table)
        except:
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
    f1_score = 0
    exact_match = 0
    arith_match = 0
    arith_total = 0
    result: List[bool] = []
    for i, d in enumerate(data):
        if not('answer' in d['pred']):
            result.append(False)
            continue

        current_f1 = 0
        current_em = 0
        if 'answer' in gold_answers[0]:
            if d['pred']['answer'] == d['gold']['answer'] and d['pred']['scale'] == d['gold']['scale']:
                current_f1 = 1.0
                current_em = 1.0
            else:
                try:
                    current_f1, current_em = evaluate_single(
                        gold_answers[i], d['pred']['answer'], d['pred']['scale'])
                except Exception as e:
                    pass

        f1_score += current_f1
        exact_match += current_em
        d['pred']['f1'] = current_f1
        d['pred']['em'] = current_em
        if current_em == 1:
            result.append(True)
        else:
            result.append(False)

        if d['gold']['type'] == 'arithmetic':
            arith_total += 1
            if abs(current_em - 1) < 1e-3:
                arith_match += 1

    print(f"evaluate {args.data_path}:")
    print(f"f1: {f1_score / len(data)}, em: {exact_match / len(data)}, arith_em: {arith_match / arith_total}")
    for d in data:
        if not('answer' in gold_answers[0]) and not(args.default_score is None):
            d['pred']['score'] = args.default_score
        else:
            d['pred']['score'] = f1_score / len(data)
        d['pred']['em_score'] = exact_match / len(data)

    return result


def process_file(generate_file_path: str, output_path: str) -> None:
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content, origin_data)

    correct_arr = evaluate(data)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example, correct in zip(data, correct_arr):
            example['correct'] = correct
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from generate.utils import evaluate_single, calc_derivation
    from generate.bart.dataset import Tabular

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--origin_data_path", "-odp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--default_score", "-ds", type=float, default=None)
    args = parser.parse_args()

    if not(args.origin_data_path is None):
        gold_answers: List[Dict[str, Any]] = []
        with open(args.origin_data_path, 'r', encoding='utf-8') as f:
            origin_data: List[Dict[str, Any]] = json.load(f)
            for d in origin_data:
                gold_answers.extend(d['questions'])

    process_file(args.data_path, args.output_path)
