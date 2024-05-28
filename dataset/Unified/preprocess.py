
import sys
import json
import random
from typing import List, Dict, Any, Tuple

sys.path.append('.')


OP_MAP: Dict[str, str] = {
    "+": "add",
    "-": "subtract",
    "*": "multiply",
    "/": "divide"
}


def transfer(derivation: str, variables: List[str]) -> Tuple[List[str], List[str]]:
    tokens = derivation.split()
    result: List[str] = []

    new_tokens: List[str] = []
    for i, t in enumerate(tokens):
        if t == '-' and (i == 0 or new_tokens[-1] in ['+', '-', '/', '*', '(']):
            new_tokens.extend(['m1', '*'])
            continue
        new_tokens.append(t)
    tokens = new_tokens

    def compare(op1: str, op2: str):
        return op1 in ["*", "/"] and op2 in ["+", "-"]

    def get_formula(num1: str, num2: str, operator: str):
        if num1 == 'm1':
            num1, num2 = num2, num1
        result.append(f"{OP_MAP[operator]} ( {num1} , {num2} )")
        return f"#{len(result) - 1}"

    def process(num_stack: List[str], op_stack: List[str]):
        operator = op_stack.pop()
        num2 = num_stack.pop()
        num1 = num_stack.pop()
        num_stack.append(get_formula(num1, num2, operator))

    idx = 0
    num_stack: List[str] = []
    op_stack: List[str] = []
    while idx < len(tokens):
        if tokens[idx].isalnum():
            num_stack.append(tokens[idx])
        elif tokens[idx] == ")":
            while op_stack[-1] != "(":
                process(num_stack, op_stack)
            op_stack.pop()
        elif not op_stack or op_stack[-1] == "(":
            op_stack.append(tokens[idx])
        elif tokens[idx] == "(" or compare(tokens[idx], op_stack[-1]):
            op_stack.append(tokens[idx])
        else:
            while op_stack and not compare(tokens[idx], op_stack[-1]):
                if op_stack[-1] == "(":
                    break
                process(num_stack, op_stack)
            op_stack.append(tokens[idx])
        idx += 1
    while op_stack:
        process(num_stack, op_stack)

    # re-shuffle variables
    variable_map: Dict[str, str] = {
        f"x{i + 1}": variables[i] for i in range(len(variables))}
    derivation_tokens = ' | '.join(result).split()
    new_variables: List[str] = []
    for i in range(len(derivation_tokens)):
        if derivation_tokens[i].startswith('x'):
            new_variables.append(variable_map[derivation_tokens[i]])
            derivation_tokens[i] = f"x{len(new_variables) - 1}"

    return ' '.join(derivation_tokens).split(' | '), new_variables


if __name__ == '__main__':
    random.seed(42)
    for mode in ['train', 'dev']:
        result: List[Dict[str, Any]] = []

        # TAT-QA
        data: List[Dict[str, Any]] = json.load(
            open(f'./dataset/TAT-QA/retrieve/{mode}.json', 'r', encoding='utf-8'))
        for d in data:
            new_questions: List[Dict[str, Any]] = []
            for q in d['questions']:
                q['uid'] = f"TAT-QA : {q['uid']}"
                if q['derivation'] and q['answer_type'] == 'arithmetic':
                    q['derivation'], q['variables'] = transfer(
                        q['derivation'], q['variables'])

                    derivation: str = ' | '.join(q['derivation'])
                    variables = q['variables']
                    answer = derivation
                    for i in range(len(variables) - 1, -1, -1):
                        answer = answer.replace(f"x{i}", variables[i])
                    q['answer'] = answer.split(' | ')

                    new_questions.append(q)
            if len(new_questions) > 0:
                d['questions'] = new_questions
                result.append(d)

        # FinQA
        data: List[Dict[str, Any]] = json.load(
            open(f'./dataset/FinQA/retrieve/{mode}.json', 'r', encoding='utf-8'))
        for d in data:
            for q in d['questions']:
                q['uid'] = f"FinQA : {q['uid']}"
                q['answer'] = [f"{x})" if not x.endswith(
                    ')') else x for x in q['origin_answer'].split('), ')]
                for i, v in enumerate(q['variables']):
                    q['variables'][i] = v.replace('const_', '')
                for i, a in enumerate(q['answer']):
                    q['answer'][i] = a.replace('const_', '')
        result.extend(data)

        # MathQA
        data: List[Dict[str, Any]] = json.load(
            open(f'./dataset/MathQA/retrieve/{mode}.json', 'r', encoding='utf-8'))
        for d in data:
            for q in d['questions']:
                q['uid'] = f"MathQA : {q['uid']}"
                for i, v in enumerate(q['variables']):
                    q['variables'][i] = v.replace(
                        'const_', '').replace('_', '.')
                for i, a in enumerate(q['answer']):
                    q['answer'][i] = a.replace('const_', '').replace('_', '.')
        result.extend(data)

        # simplify
        def simplify(s: str) -> str:
            op, other = s.split('(')
            op = op.strip()
            args = [x.strip() for x in other.strip(')').split(', ')]
            return f"{op} ( {' , '.join(args)} )"

        for d in result:
            new_questions = []
            for q in d['questions']:
                try:
                    q['answer'] = [simplify(x) for x in q['answer']]
                    q['derivation'] = [simplify(x)
                                       for x in q['derivation']]
                    new_questions.append({
                        "uid": q['uid'],
                        "question": q['question'],
                        "answer": q['answer'],
                        "derivation": q['derivation'],
                        "variables": q['variables'],
                        "rel_paragraphs": q['rel_paragraphs'] if 'rel_paragraphs' in q else []
                    })
                except:
                    print(q['answer'])
            if new_questions:
                d['questions'] = [q for q in new_questions]

        random.shuffle(result)
        with open(f'./dataset/Unified/retrieve/{mode}.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
