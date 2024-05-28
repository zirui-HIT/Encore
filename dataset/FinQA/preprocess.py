import sys
import json
from copy import deepcopy
from typing import Dict, List, Any, Tuple

sys.path.append('.')

OPERATION_MAP = {
    'add': 'add',
    'minus': 'subtract',
    'multiply': 'multiply',
    'divide': 'divide',
    'average': 'table_average',
    'sum': 'table_sum',
    'min': 'table_min',
    'max': 'table_max',
    'compare_larger': 'greater',
    'exp': 'exp'
}


def clean_table(table: List[List[str]]) -> List[List[str]]:
    for i in range(1, len(table)):
        for j in range(1, len(table[i])):
            if not ('(' in table[i][j]):
                continue
            if len(table[i][j].split('(')) != 2:
                continue
            table[i][j] = table[i][j].split('(')[0].strip()
    return table


def decompose(steps: List[Dict[str, str]], table: List[List[str]]) -> Tuple[List[str], List[str]]:
    derivations: List[str] = []
    variables: List[str] = []

    def update_arg(x: str) -> str:
        t: str = ''
        if x.startswith('#'):
            t = x
        elif x.startswith('const_'):
            t = x.lstrip('const_')
            variables.append(t)
            t = f"x{len(variables) - 1}"
        else:
            t = f"x{len(variables)}"
            variables.append(x)
        return t

    for s in steps:
        op = OPERATION_MAP[s['op'][:-3]]
        if op in ['add', 'subtract', 'multiply', 'divide', 'greater', 'exp']:
            arg1 = update_arg(s['arg1'])
            arg2 = update_arg(s['arg2'])
            derivations.append(f"{op} ( {arg1} , {arg2} )")
        else:
            derivations.append(f"{op} ( x{len(variables)} , {s['arg2']} )")
            variables.append(s['arg1'])

    for i, v in enumerate(variables):
        if v.startswith('.'):
            variables[i] = f"0{v}"

    return derivations, variables


def fill(derivations: List[str], variables: List[str]) -> List[str]:
    count = 0
    result: List[str] = deepcopy(derivations)
    for i in range(len(result)):
        while f"x{count}" in result[i]:
            result[i] = result[i].replace(f"x{count}", variables[count])
            count += 1
    return result


if __name__ == '__main__':
    for part in ['dev', 'train', 'test']:
        if part == 'test':
            part = f'private_{part}'
        with open(f'/users10/dzrwang/HybridQA/dataset/FinQA/{part}.json', 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)

        result: List[Dict[str, Any]] = []
        for i, d in enumerate(data):
            instance: Dict[str, Any] = {
                "table": {},
                "paragraphs": [],
                "questions": []
            }

            instance['table']['uid'] = f"{part}-{i}"
            instance['table']['table'] = clean_table(d['table'])
            for j, p in enumerate(d['pre_text'] + d['post_text']):
                instance['paragraphs'].append({
                    "uid": f"{part}-{i}-{j}",
                    "order": j + 1,
                    "text": p
                })

            question: Dict[str, Any] = {}
            question['uid'] = d['id']
            question['order'] = 0
            question['question'] = d['qa']['question']
            if not ('test' in part):
                question['derivation'], question['variables'] = decompose(
                    d['qa']['steps'], instance['table']['table'])
                question['answer'] = fill(
                    question['derivation'], question['variables'])
                question['origin_answer'] = d['qa']['program']
                question['origin_answer_value'] = d['qa']['exe_ans']
                question['rel_paragraphs'] = [
                    str(x + 1) for x in d['qa']['ann_text_rows']]
                question['rel_table_cols'] = d['qa']['ann_table_rows']
            instance['questions'].append(question)

            result.append(instance)

        part = part.split('_')[-1]
        with open(f"./dataset/FinQA/origin/{part}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
