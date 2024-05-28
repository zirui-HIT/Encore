import sys

from copy import deepcopy
from word2number.w2n import word_to_num
from typing import Any, Dict, List, Tuple

sys.path.append('.')


class Tabular(object):
    def __init__(self, table: List[List[str]], pointer_flag: bool = True):
        self.table = deepcopy(table)
        self.pointer_flag = pointer_flag
        self.col_number = len(table)
        self.row_number = len(table[0])

    def column_linear(self, with_rank: bool = True) -> str:
        from generate.utils import numerical, convert_to_tuple_list

        def oridinal(number: str) -> str:
            if number[-1] == '1':
                return number + 'st'
            if number[-1] == '2':
                return number + 'nd'
            if number[-1] == '3':
                return number + 'rd'
            return number + 'th'

        all_rows: List[str] = []
        for i in range(self.col_number):
            all_number_with_rank = []
            if i != 0 and self.all_number(i):
                all_number = [numerical(x) for x in self.table[i][1:]]
                all_number_with_rank = convert_to_tuple_list(all_number)

            all_row_cells: List[str] = []
            for j in range(self.row_number):
                cell = str(self.table[i][j])
                if not with_rank:
                    all_row_cells.append(cell)
                    continue
                if self.pointer_flag and (i == 0 or j == 0):
                    cell = f"{self.mark(i, j)} @ {cell}"
                if j != 0 and all_number_with_rank:
                    cell = f"{oridinal(str(all_number_with_rank[j - 1][1] + 1))} @ {cell}"
                all_row_cells.append(cell)
            all_rows.append(' | '.join(all_row_cells))
        return ' | NC '.join(all_rows)

    def linear(self, kind: str, with_rank: bool = True) -> str:
        if kind == 'column':
            return self.column_linear(with_rank)
        raise Exception('illegal linear type')

    def get_item(self, i: int, j: int) -> str:
        if i == 0 or j == 0:
            return self.table[i][j].split(' : ')[-1]
        return self.table[i][j]

    def get_item_with_coordinate(self, i: int, j: int) -> Tuple[str, str, str]:
        col_name = self.get_item(i, 0)
        row_name = self.get_item(0, j)
        cell_value = self.get_item(i, j)
        return col_name, row_name, cell_value

    def all_number(self, col_idx: int) -> bool:
        from generate.utils import calc_derivation

        if col_idx >= self.col_number:
            return False
        for x in self.table[col_idx][1:]:
            flag = True
            try:
                word_to_num(x)
            except:
                flag = False
            try:
                calc_derivation(x)
            except:
                if not flag:
                    return False
        return True

    def index(self, value: str) -> str:
        if not self.pointer_flag:
            return value

        def same(a: str, b: str) -> bool:
            a, b = a.lower(), b.lower()
            if a == b:
                return True

            def clean(x: str) -> str:
                for t in ['$', '%', 'thousand', 'million', 'billion', 'percent']:
                    x = x.replace(t, '')
                return x.strip(' : ').strip('.').strip()
            a, b = clean(a), clean(b)

            def equal(a: str, b: str) -> bool:
                if a == b:
                    return True
                if a.replace('(', '').replace(')', '').strip() == b.replace('(', '').replace(')', '').strip():
                    return True
                if b in ['2015', '2016', '2017', '2018', '2019', '2020'] and b in a:
                    return True
                if all(x.isalpha() or x.isspace() for x in a[:-1]):
                    for x in ['1', '2', '3', '4', '5']:
                        if a.endswith(x) and a[:-1].strip() == b:
                            return True
                for x in ['1', '2', '3', '4', '5', '6']:
                    if a.endswith(f'({x})') and a[:-3].strip() == b:
                        return True
                    if a.endswith(f'{x})') and a[:-2].strip() == b:
                        return True
                for x in ['¹', '²', '³', '⁴', '⁵', '₁', '₂', '₃', '₄', '₅']:
                    if a.endswith(x) and a[:-1].strip() == b:
                        return True
                return a == b

            for x in a.split(' : '):
                if equal(x, b):
                    return True
            return False

        value = value.strip()
        for i in range(self.col_number):
            for j in range(self.row_number):
                if same(self.table[i][j], value):
                    return self.locate(i, j)

        return value

    def get(self, mark: str) -> str:
        return self.get_coordinate(mark)[2]

    def get_coordinate(self, mark: str) -> Tuple[str, str, str]:
        try:
            mark = ' , '.join([x.strip() for x in mark.split(',')])
            _, col_idx, _, row_idx, _ = mark.split()
            col_idx = int(col_idx.strip('col')) if col_idx != 'none' else 0
            row_idx = int(row_idx.strip('row')) if row_idx != 'none' else 0
            return self.table[0][row_idx], self.table[col_idx][0], self.table[col_idx][row_idx]
        except:
            return None, None, mark

    def locate(self, i: int, j: int) -> str:
        column = self.mark(i, 0) if i != 0 else 'none'
        row = self.mark(0, j) if j != 0 else 'none'
        return "{ " + f"{column} , {row}" + " }"

    def mark(self, i: int, j: int) -> str:
        if i == 0 and j == 0:
            return ""
        if i == 0:
            return f"row{j}"
        if j == 0:
            return f"col{i}"
        raise Exception(f'illegal position {i} , {j}')
