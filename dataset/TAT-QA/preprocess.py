import sys
import json

from typing import List, Dict, Any, Tuple

sys.path.append('.')


def process_table(table: List[List[str]]) -> List[List[str]]:
    from generate.utils import clean_str

    # clear table
    table = [[clean_str(x) for x in c] for c in table]
    table = [c for c in table if len([x for x in c if x]) > 0]

    ignore_column_id: List[int] = []
    # remove all same column
    for i, column in enumerate(table):
        cells = list(set([x for x in column if x]))
        if len(cells) == 1:
            table[i] = [cells[0] if i == 0 else "" for i in range(len(column))]
    # remove district column
    for i, column in enumerate(table):
        cells = [x for x in column if x]
        if len(cells) == 1:
            if i != len(table) - 1:
                table[i +
                      1][0] = f"{cells[0].strip(':')} : {table[i + 1][0]}".strip(' : ')
            ignore_column_id.append(i)
    table = [c for i, c in enumerate(table) if not (i in ignore_column_id)]

    # remove empty row
    ignore_row_id: List[int] = []
    for i in range(len(table[0])):
        row_cells: List[str] = [table[j][i]
                                for j in range(len(table)) if table[j][i]]
        if not row_cells:
            ignore_row_id.append(i)
    table = [[table[i][j] for j in range(len(table[0])) if not (
        j in ignore_row_id)] for i in range(len(table))]

    return table


def process_paragraph(para: str) -> str:
    # para = para.replace('\n', ' ').strip()
    return para


def update_derivation(s: str, number_char: List[str] = ['%', '.', ',', 'b', 'm', 't']) -> Tuple[str, List[str]]:
    s = ''.join(s.split())
    s = s.replace('$', '')
    s = s.replace('[', '(').replace(']', ')')
    s = s.replace('billions', 'b').replace('billion', 'b')
    s = s.replace('millions', 'm').replace('million', 'm')
    s = s.replace('thousands', 't').replace('thousand', 't')

    i = 0
    derivation: List[str] = []
    variables: List[str] = []
    while i < len(s):
        if s[i].isdigit() or s[i] in number_char:
            start = i
            while i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] in number_char):
                i += 1
            current_number = s[start: i + 1]

            if derivation and derivation[-1] == '/' and current_number in [str(x) for x in range(8)]:
                derivation.append(current_number)
            elif not derivation or not (derivation[-1].startswith('x')):
                variables.append(current_number)
                derivation.append(f"x{len(variables)}")
        elif s[i] in ['+', '-', '*', '/', '(', ')']:
            derivation.append(s[i])
        i += 1

    # add scale
    variables = [x.replace('b', 'billion').replace(
        'm', 'million').replace('t', 'thousand') for x in variables]

    # fix negative error
    i = 0
    new_derivation: List[str] = []
    while i < len(derivation):
        if derivation[i] == '-' and derivation[i-1] in ['+', '-', '*', '/'] and derivation[i+1].startswith('x'):
            new_derivation.extend(['(', '-', derivation[i+1], ')'])
            i += 1
        else:
            new_derivation.append(derivation[i])
        i += 1
    derivation = new_derivation

    # sample derivation
    i = 0
    new_derivation: List[str] = []
    while i < len(derivation):
        if not (i in [0, len(derivation) - 1]) and derivation[i].startswith('x') and derivation[i-1] == '(' and derivation[i+1] == ')':
            new_derivation.pop(-1)
            new_derivation.append(derivation[i])
            i += 2
        else:
            new_derivation.append(derivation[i])
            i += 1
    if new_derivation[0] == '(' and new_derivation[-1] == ')' and new_derivation.count('(') == 1:
        new_derivation = new_derivation[1:-1]
    derivation = new_derivation

    return " ".join(derivation), variables


def preprocess_dataset(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    derivations: Dict[str, int] = {}
    for d in data:
        for p in d['paragraphs']:
            p['text'] = process_paragraph(p['text'])
        d['table']['table'] = process_table(d['table']['table'])
        for q in d['questions']:
            if 'answer_type' in q and q['answer_type'] == 'arithmetic':
                q['origin_derivation'] = q['derivation']
                q['derivation'], q['variables'] = update_derivation(
                    q['derivation'])
                derivations[q['derivation']] = derivations[q['derivation']
                                                           ] + 1 if q['derivation'] in derivations else 1
            if 'answer_from' in q:
                table = Tabular(d['table']['table'])
                if q['answer_type'] == 'arithmetic':
                    variable_list = q['variables']
                elif 'span' in q['answer_type']:
                    variable_list = q['answer']
                elif q['answer_type'] == 'count':
                    variable_list = q['derivation'].split('##')
                q['rel_table_cols'] = []
                if 'table' in q['answer_from']:
                    q['rel_table_cols'] = [x.split(',')[0].strip().strip('{ col') for x in [table.index(y.strip()) for y in variable_list]]
                    q['rel_table_cols'] = list(set([int(x) for x in q['rel_table_cols'] if x.isnumeric()]))
    return data


if __name__ == '__main__':
    from generate.bart.dataset import Tabular

    for mode in ['train', 'dev', 'test']:
        with open(f'/users10/dzrwang/HybridQA/dataset/TATQA/{mode}.json', "r", encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)
        data = preprocess_dataset(data)

        with open(f'./dataset/TAT-QA/origin/{mode}.json', "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"preprocess {mode} done")
