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
            source = f"<Q> what is the name of {table.mark(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<A> {table.get_item(i, 0)}"
            result.append((source, target))
        for i in range(1, table.row_number):
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

            # sum
            source = f"<Q> what is the sum of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<D> table_sum ( x0 , none ) | <V> {table.locate(i, 0)} | <A> table_sum ( {table.get_item(i, 0)} , none )"
            result.append((source, target))

            # avg
            source = f"<Q> what is the average of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<D> table_average ( x0, none ) | <V> {table.locate(i, 0)} | <A> table_average ( {table.get_item(i, 0)} , none )"
            result.append((source, target))

            # max
            source = f"<Q> what is the maximum of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<D> table_max ( x0, none ) | <V> {table.locate(i, 0)} | <A> table_max ( {table.get_item(i, 0)} , none )"
            result.append((source, target))

            # min
            source = f"<Q> what is the minimum of {table.get_item(i, 0)} ? | <T> {table.linear('column')}"
            target = f"<D> table_min ( x0, none ) | <V> {table.locate(i, 0)} | <A> table_min ( {table.get_item(i, 0)} , none )"
            result.append((source, target))
        random.shuffle(result)
        return result

    result = table_cell_location(table) + table_column_calculation(table)
    return result


if __name__ == '__main__':
    from generate.bart.dataset import Tabular
    from generate.utils import duplicate_remove

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--data_size", "-ds", type=int)
    parser.add_argument("--table_pointer", "-tp", action="store_true")
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
