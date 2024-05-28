import json

from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Any, Tuple


def decompose(formula: str) -> Tuple[List[str], List[str]]:
    formula = formula.replace(
        '(', ' ( ').replace(')', ' ) ').replace(', ', ' , ')
    spans = [x for x in formula.split() if x and x != ',']

    stack: List[Tuple[str, int]] = []
    tree: List[List[int]] = [[] for _ in range(len(spans))]
    for i, s in enumerate(spans):
        if s == ')':
            idx: List[int] = []
            while stack[-1][0] != '(':
                idx.append(stack[-1][1])
                stack = stack[:-1]
            stack = stack[:-1]
            tree[stack[-1][1]] = [x for x in reversed(idx)]
        else:
            stack.append((s, i))

    derivations: List[str] = []
    variables: List[str] = []

    def DFS(x: int) -> str:
        if not tree[x]:
            alias = f"x{len(variables)}"
            variables.append(spans[x])
            return alias

        param: List[str] = []
        for t in tree[x]:
            param.append(DFS(t))
        derivations.append(f"{spans[x]} ( {' , '.join(param)} )")
        return f"#{len(derivations) - 1}"
    DFS(0)

    # reorder veriables
    new_variables: List[str] = []
    for i, d in enumerate(derivations):
        tokens = d.split()
        for j, token in enumerate(tokens):
            if token.startswith('x'):
                new_variables.append(variables[int(token.strip('x'))])
                tokens[j] = f"x{len(new_variables) - 1}"
        derivations[i] = " ".join(tokens)
    variables = new_variables

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
        with open(f'./dataset/MathQA/origin/{part}.json', 'r', encoding='utf-8') as f:
            data: List[Dict[str, str]] = json.load(f)

        result: List[Dict[str, Any]] = []
        length_count: Dict[int, int] = {}
        for i, d in tqdm(enumerate(data)):
            instance: Dict[str, Any] = {
                "table": {},
                "paragraphs": [],
                "questions": []
            }

            question: Dict[str, Any] = {}
            question['uid'] = f"{part}-{i}"
            question['order'] = 0
            question['question'] = d['Problem']
            question['derivation'], question['variables'] = decompose(
                d['annotated_formula'])
            question['answer'] = fill(
                question['derivation'], question['variables'])
            instance['questions'].append(question)

            length = len(question['answer'])
            length_count[length] = length_count[length] + \
                1 if length in length_count else 1

            if part == 'train' and len(question['answer']) > 10:
                continue
            result.append(instance)

        with open(f"./dataset/MathQA/retrieve/{part}.json", 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
