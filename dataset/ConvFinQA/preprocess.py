import sys
sys.path.append('.')

import re
import json

from copy import deepcopy
from typing import Dict, List, Any, Tuple
from generate.utils import calc_derivation


OPERATION_MAP = {
    'add': 'add',
    'minus': 'subtract',
    'subtract': 'subtract',
    'multiply': 'multiply',
    'divide': 'divide',
    'average': 'table_average',
    'sum': 'table_sum',
    'min': 'table_min',
    'max': 'table_max',
    'compare_larger': 'greater',
    'greater': 'greater',
    'exp': 'exp'
}


def clean_table(table: List[List[str]]) -> List[List[str]]:
    for i in range(1, len(table)):
        for j in range(1, len(table[i])):
            if not('(' in table[i][j]):
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


def program_to_steps(program: str) -> List[Dict[str, str]]:
    functions = []
    expression_pattern = re.compile(r'(\w+)\((.*?), (.*?)\)')
    matches = expression_pattern.findall(program)
    result_map: Dict[str, float] = {}

    for i, match in enumerate(matches):
        functions.append({
            "op": f"{match[0]}0-0",
            "arg1": match[1],
            "arg2": match[2],
        })

        def process_arg(n: str) -> float:
            n = n.strip('const_')
            if n.startswith('#'):
                n = result_map[n]
            if n == 'm1':
                n = -1
            return n
        
        arg1 = process_arg(match[1])
        arg2 = process_arg(match[2])
        op = match[0]
        if op == 'subtract':
            op = "-"
        elif op == 'add':
            op = "+"
        elif op == 'divide':
            op = "/"
        elif op == "multiply":
            op = "*"
        if op == 'greater':
            functions[-1]['res'] = 'yes' if float(arg1) > float(arg2) else 'no'
        elif op =='exp':
            functions[-1]['res'] = float(arg1) ** float(arg2)
        else:
            functions[-1]['res'] = calc_derivation(f"({arg1}) {op} ({arg2})")[0]
        result_map[f"#{i}"] = functions[-1]['res']

    return functions


if __name__ == '__main__':
    for part in ['dev_turn', 'train_turn', 'test_turn_private']:
        with open(f'/users10/dzrwang/Numerical/dataset/ConvFinQA/{part}.json', 'r', encoding='utf-8') as f:
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
            question['question'] = ' | '.join(d['annotation']['cur_dial'])
            if not 'test' in part:
                question['derivation'], question['variables'] = decompose(
                    program_to_steps(d['annotation']['cur_program']), instance['table']['table'])
                question['answer'] = fill(
                    question['derivation'], question['variables'])
                question['origin_answer'] = d['annotation']['cur_program']
                question['origin_answer_value'] = d['annotation']['exe_ans']
                if 'qa_0' in d and 'qa_1' in d:
                    d['qa'] = {}
                    d['qa']['ann_text_rows'] = list(set(d['qa_0']['ann_text_rows'] + d['qa_1']['ann_text_rows']))
                question['rel_paragraphs'] = [
                    str(x + 1) for x in d['qa']['ann_text_rows']]
            instance['questions'].append(question)
            result.append(instance)

        with open(f"./dataset/ConvFinQA/origin/{part.split('_')[0]}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
