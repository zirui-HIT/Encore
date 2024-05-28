import re
import sys
import json
import argparse
from re import RegexFlag
from copy import deepcopy
from typing import Any, Dict, List, Union
from math import log2, sqrt, sin, cos, tan, factorial, gcd, floor, pi

sys.path.append(".")


OPERATION_MAP: Dict[str, Any] = {
    "add": (lambda x: x[0] + x[1]),
    "subtract": (lambda x: x[0] - x[1]),
    "multiply": (lambda x: x[0] * x[1]),
    "divide": (lambda x: x[0] / x[1]),
    "log": (lambda x: log2(x[0])),
    "sqrt": (lambda x: sqrt(x[0])),
    "factorial": (lambda x: factorial(int(x[0])) if x[0] < 64 else 1),
    "gcd": (lambda x: gcd(int(x[0]), int(x[1]))),
    "lcm": (lambda x: int(x[0]) * int(x[1]) / OPERATION_MAP["gcd"](x)),
    "power": (lambda x: x[0] ** x[1]),
    "reminder": (lambda x: x[0] % x[1]),
    "negate": (lambda x: -x[0]),
    "inverse": (lambda x: 1 / x[0]),
    "round": (lambda x: round(x[0])),
    "max": max,
    "min": min,
    "floor": (lambda x: floor(x[0])),
    "sine": (lambda x: sin(x[0])),
    "cosine": (lambda x: cos(x[0])),
    "tangent": (lambda x: tan(x[0])),
    "triangle_area": (lambda x: x[0] * x[1] / 2),
    "rectangle_area": (lambda x: x[0] * x[1]),
    "square_area": (lambda x: x[0] * x[0]),
    "surface_rectangular_prism": (lambda x: 2 * (x[0] * x[1] + x[0] * x[2] + x[1] * x[2])),
    "volume_rectangular_prism": (lambda x: x[0] * x[1] * x[2]),
    "rectangle_perimeter": (lambda x: 2 * (x[0] + x[1])),
    "sum": sum,
    "average": (lambda x: sum(x) / len(x)),
    "greater": (lambda x: x[0] > x[1]),
    "circle_area": (lambda x: pi * x[0] * x[0]),
    "exp": (lambda x: x[1] ** x[0]),
    "choose": (lambda x: OPERATION_MAP["factorial"](x[0]) / OPERATION_MAP["factorial"](x[1]) / OPERATION_MAP["factorial"](x[0] - x[1])),
    "circumface": (lambda x: 2 * pi * x[0]),
    "permutation": (lambda x: OPERATION_MAP["factorial"](x[0]) / OPERATION_MAP["factorial"](x[1])),
    "rhombus_perimeter": (lambda x: 4 * x[0]),
    "surface_cube": (lambda x: 6 * x[0] * x[0]),
    "volume_cube": (lambda x: x[0] ** 3),
    "speed": (lambda x: x[0] / x[1])
}


def process_const(c: str) -> float:
    c = c.strip('const_')
    if c == 'pi':
        return pi
    elif c == 'deg_to_rad':
        raise Exception('illegal const')
    else:
        c = c.replace('_', '.')
        return float(c)


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


def eval_calculate(program: List[str], table) -> Union[float, bool]:
    from generate.bart.dataset import Tabular

    table: Tabular = table
    values: List[float] = []
    for p in program:
        p = p.lower()
        p = p.replace(' , row', ',row')
        if not p.startswith('table'):
            p = p.replace(' , none }', ',none }')

        operation, other = p.split('(')
        param, other = other.split(')')
        params = [x.strip() for x in param.split(' , ')]
        operation = operation.strip()

        params_float: List[float] = []
        for x in params:
            if operation.startswith('table'):
                params_float.append(x)
            elif x.startswith('const'):
                params_float.append(process_const(x))
            elif x.startswith('#'):
                params_float.append(values[int(x.strip('#'))])
            elif 'col' in x or 'row' in x:
                x = x.replace(',', ' , ')
                params_float.append(calc_derivation(table.get(x))[0])
            elif x == 'm1':
                params_float.append(-1)
            elif x == 'pi':
                params_float.append(pi)
            else:
                params_float.append(calc_derivation(x)[0])

        if operation.startswith('table_'):
            idx = table.index(params_float[0])
            idx = idx.strip('{ }')
            idx = idx.split(' , ')[0]
            col_idx = int(idx.strip('col'))
            params_float = [calc_derivation(table.table[col_idx][i])[
                0] for i in range(1, table.row_number)]
            operation = operation[6:]
        values.append(OPERATION_MAP[operation](params_float))

    return values[-1]


def eval_match(gold_program: List[str], pred_program: List[str], table) -> bool:
    if ' '.join(gold_program) == ' '.join(pred_program):
        return True
    try:
        x = eval_calculate(gold_program, table)
        y = eval_calculate(pred_program, table)
        if x is None:
            return False
        if isinstance(x, bool) and isinstance(y, bool):
            if x and y:
                return True
            if not x and not y:
                return True
            return False
        return abs(x - y) < 1e-4
    except Exception as e:
        # print(f"{e} : {' | '.join(pred_program)}")
        return False


def process_file(generate_file_path: str, output_path: str) -> None:
    from generate.bart.dataset import Tabular

    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content)

    match_count: Dict[str, int] = {
        'Unified': 0
    }
    total_count: Dict[str, int] = {
        'Unified': 0
    }
    for i, d in enumerate(data):
        d['correct'] = False
        if not('answers' in d['gold']) or not ('answers' in d['pred']):
            continue

        db = db_id[i]
        d['db_id'] = db
        gold_answer = answers[i]
        table = Tabular(tables[i]) if tables[i] else None

        if eval_match(gold_answer, d['pred']['answers'], table):
            match_count['Unified'] += 1
            match_count[db] = match_count[db] + 1 if db in match_count else 1
            d['correct'] = True
        total_count['Unified'] += 1
        total_count[db] = total_count[db] + 1 if db in total_count else 1

        try:
            d['gold']['value'] = str(
                eval_calculate(d['gold']['answers'], table))
        except:
            d['gold']['value'] = None
        try:
            d['pred']['value'] = str(
                eval_calculate(d['pred']['answers'], table))
        except:
            d['pred']['value'] = None

    for db in match_count:
        print(f"{db} : {match_count[db] / total_count[db]}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from generate.utils import calc_derivation

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str,
                        default='./generate/seq2seq/checkpoint/Unified/finetune/BART.base/generate-valid.txt')
    parser.add_argument("--origin_data_path", "-odp", type=str,
                        default='./dataset/Unified/retrieve/dev.json')
    parser.add_argument("--output_path", "-op", type=str,
                        default='./generate/vote/dataset/Unified/dev.BART.base.json')
    args = parser.parse_args()

    db_id: List[str] = []
    tables: List[List[List[str]]] = []
    answers: List[List[str]] = []
    origin_data = json.load(open(args.origin_data_path, 'r', encoding='utf-8'))
    for d in origin_data:
        for q in d['questions']:
            db_id.append(q['uid'].split(' : ')[0])
            tables.append(d['table']['table']
                          if 'table' in d['table'] else None)

            answer = ' | '.join(q['derivation'])
            for i in range(len(q['variables']) - 1, -1, -1):
                answer = answer.replace(f"x{i}", q['variables'][i])
            answers.append(answer.split(' | '))

    process_file(args.data_path, args.output_path)
