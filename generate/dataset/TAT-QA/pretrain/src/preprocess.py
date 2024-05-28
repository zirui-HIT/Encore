import os
import sys
import json
import random
import argparse
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

sys.path.append('.')


def generate_table_data(table) -> List[Tuple[str, str]]:
    from generate.bart.dataset import Tabular
    table: Tabular = table

    def table_cell_location(table: Tabular) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        for i in range(1, table.col_number):
            for j in range(1, table.row_number):
                col_name, row_name, cell_value = table.get_item_with_coordinate(
                    i, j)
                if not col_name or not row_name or cell_value in ['', '-']:
                    continue
                source = f"<Q> what is {table.locate(i, j)} ? | <T> {table.linear('column')}"
                target = f"<A> {cell_value}"
                result.append((source, target))
        for i in range(1, table.col_number):
            if not table.get_item(i, 0):
                continue
            source = f"<Q> what is the name of {table.mark(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<A> {table.get_item(i, 0)}"
            result.append((source, target))
        for i in range(1, table.row_number):
            if not table.get_item(0, i):
                continue
            source = f"<Q> what is the name of {table.mark(0, i)} ? | <T> {table.linear('column')}"
            target = f"<A> {table.get_item(0, i)}"
            result.append((source, target))
        random.shuffle(result)
        return result

    def table_column_calculation(table: Tabular) -> List[Tuple[str, str]]:
        result: List[Tuple[str, str]] = []
        for i in range(1, table.col_number):
            if not table.get_item(i, 0) or not table.all_number(i):
                continue
            all_number: List[Tuple[int, str]] = [
                (j, x) for j, x in enumerate(table.table[i]) if j != 0 and x]

            # sum
            derivation = ' + '.join(
                [f'x{j + 1}' for j in range(len(all_number))])
            variables = [table.locate(i, x[0]) for x in all_number]
            source = f"<Q> what is the sum of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<D> {derivation} | <V> {' | '.join(variables)} | <A> {variable_replace(derivation, variables)}"
            result.append((source, target))

            # avg
            derivation = f"( {' + '.join([f'x{j + 1}' for j in range(len(all_number))])} ) / {len(all_number)}"
            variables = [table.locate(i, x[0]) for x in all_number]
            source = f"<Q> what is the average of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<D> {derivation} | <V> {' | '.join(variables)} | <A> {variable_replace(derivation, variables)}"
            result.append((source, target))

            # max
            max_value = max(all_number, key=(lambda x: calc_derivation(x[1])))
            source = f"<Q> what is the maximum of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<V> {table.locate(i, max_value[0])} | <A> {max_value[1]}"
            result.append((source, target))

            # min
            min_value = min(all_number, key=(lambda x: calc_derivation(x[1])))
            source = f"<Q> what is the minimum of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<V> {table.locate(i, min_value[0])} | <A> {min_value[1]}"
            result.append((source, target))
        random.shuffle(result)
        return result

    def table_expand_location(table: Tabular) -> List[Tuple[str, str]]:
        for i in range(table.col_number):
            if ':' in table.table[i][0]:
                return []
        for i in range(table.row_number):
            if not table.table[0][i]:
                return []

        def expand_row(table: List[List[str]], row_number: int) -> List[List[str]]:
            result: List[List[str]] = [[t[0]] for t in table]
            for i in range(len(table)):
                for j in range(1, len(table[0])):
                    for k in range(row_number):
                        if not i and not k:
                            result[i].append(table[i][j])
                        elif not i and k:
                            result[i].append('')
                        else:
                            result[i].append(random.randint(1, 1000))
            return result

        def expand_column1(table: List[List[str]], col_number: int) -> Tuple[List[List[str]], List[str]]:
            col_names: List[str] = [
                f"column{random.randint(1, 100)}" for _ in range(col_number)]

            result: List[List[str]] = [table[0]]
            for i in range(col_number):
                result.extend([[table[k][j] if not j else random.randint(
                    1, 1000) for j in range(len(table[0]))] for k in range(1, len(table))])
                result[1 + i * (len(table) - 1)
                       ][0] = f"{col_names[i]} : {result[1 + i * (len(table) - 1)][0]}"
            return result, col_names

        def expand_column2(table: List[List[str]], col_number: int) -> Tuple[List[List[str]], List[str]]:
            col_names: List[str] = [
                f"column{random.randint(1, 100)}" for _ in range(col_number)]

            result: List[List[str]] = []
            for i in range(col_number):
                result.extend([[table[k][j] if not k or not j else random.randint(
                    1, 1000) for j in range(len(table[0]))] for k in range(len(table))])
                result[i * len(table)
                       ][0] = f"{col_names[i]} : {result[i * len(table)][0]}"
            return result, col_names

        row_number = random.randint(2, 3)
        col_number = random.randint(2, 3)
        result: List[Tuple[str, str]] = []

        tab, col_names = expand_column1(
            expand_row(table.table, row_number), col_number)
        new_table = Tabular(tab, args.table_pointer)
        for i in range(1, new_table.col_number):
            source = f"<Q> what is {new_table.mark(i, 0)} belong to ? | <T> {new_table.linear('column')}"
            target = f"<A> {col_names[(i - 1) // (table.col_number - 1)]}"
            result.append((source, target))

            for j in range(1, new_table.row_number):
                source = f"<Q> what is the {new_table.locate(i, j)} belong to ? | <T> {new_table.linear('column')}"
                target = f"<A> {new_table.get_item(0, 1 + (j - 1) // row_number * row_number)} | {col_names[(i - 1) // (table.col_number - 1)]}"
                result.append((source, target))

        tab, col_names = expand_column2(
            expand_row(table.table, row_number), col_number)
        new_table = Tabular(tab, args.table_pointer)
        for i in range(1, new_table.col_number):
            if ':' in new_table.table[i][0]:
                continue
            source = f"<Q> what is {new_table.mark(i, 0)} belong to ? | <T> {new_table.linear('column')}"
            target = f"<A> {col_names[i // table.col_number]}"
            result.append((source, target))

            for j in range(1, new_table.row_number):
                source = f"<Q> what is the {new_table.locate(i, j)} belong to ? | <T> {new_table.linear('column')}"
                target = f"<A> {new_table.get_item(0, 1 + (j - 1) // row_number * row_number)} | {col_names[i // table.col_number]}"
                result.append((source, target))

        random.shuffle(result)
        return result

    tcl_result = table_cell_location(table)
    tcc_result = table_column_calculation(table)
    tel_result = table_expand_location(
        table)[:(len(tcl_result) + len(tcc_result)) // 2]
    if args.not_include == 'TCT':
        return tcl_result + tel_result
    elif args.not_include == 'HTP':
        return tcl_result + tcc_result
    elif args.not_include == 'ALP':
        return tcc_result + tel_result
    elif args.not_include == 'None':
        return tcc_result + tel_result + tcl_result


if __name__ == '__main__':
    from generate.bart.dataset import Tabular
    from generate.utils import duplicate_remove, calc_derivation, variable_replace

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--data_size", "-ds", type=int)
    parser.add_argument("--table_pointer", "-tp", action="store_true")
    parser.add_argument("--not_include", "-ni",
                        choices=['TCT', 'HTP', 'ALP', 'None'], default='None')
    args = parser.parse_args()
    random.seed(42)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    result: Dict[str, List[Tuple[str, str]]] = {}
    for part in ['dev', 'train']:
        with open(f"{args.data_path}/{part}.json", 'r', encoding='utf-8') as f:
            data: List[Dict[str, Any]] = json.load(f)

        result[part] = []
        for d in tqdm(data, desc=part):
            result[part] += generate_table_data(
                Tabular(d['table']['table'], args.table_pointer))

        if part == 'train':
            result[part] = duplicate_remove(result[part])[:args.data_size]
        else:
            result[part] = duplicate_remove(result[part])[:2000]

        with open(f"{args.output_path}/{part}.src", "w", encoding='utf-8') as f:
            f.write("\n".join([r[0] for r in result[part]]))
        with open(f"{args.output_path}/{part}.tgt", "w", encoding='utf-8') as f:
            f.write("\n".join([r[1] for r in result[part]]))

    print(f"preprocess {args.data_path} done")
