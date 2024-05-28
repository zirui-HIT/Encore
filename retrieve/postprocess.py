import json
import random
import argparse

from tqdm import tqdm
from typing import List, Dict, Any


def get_separated_result(
    origin_data: List[Dict[str, Any]],
    text_top_k: int,
    table_top_k: int,
    golden_flag: bool
):
    em_count = 0
    text_em_count = 0
    table_em_count = 0
    total_count = 0
    for d in tqdm(origin_data):
        for q in d['questions']:
            # filter
            pred_text_idx = []
            pred_table_idx = []
            data[total_count]['ctxs'] = sorted(
                data[total_count]['ctxs'], key=(lambda x: x['score']), reverse=True)
            for x in data[total_count]['ctxs']:
                if x['idx'].startswith('text'):
                    pred_text_idx.append(str(int(x['idx'].strip('text_')) + 1))
                if x['idx'].startswith('table'):
                    pred_table_idx.append(int(x['idx'].strip('table_')))
            pred_text_idx = pred_text_idx[:text_top_k]
            pred_table_idx = pred_table_idx[:table_top_k]
            all_pred_idx = [f"text_{x}" for x in pred_text_idx] + \
                [f"table_{x}" for x in pred_table_idx]

            # evaluate
            text_match = not ('rel_paragraphs' in q)
            table_match = not ('rel_table_cols' in q)
            if 'rel_paragraphs' in q and len([x for x in q['rel_paragraphs'] if x in pred_text_idx]) == len(q['rel_paragraphs']):
                text_match = True
                text_em_count += 1
            if 'rel_table_cols' in q and len([x for x in q['rel_table_cols'] if x in pred_table_idx]) == len(q['rel_table_cols']):
                table_match = True
                table_em_count += 1
            if text_match and table_match:
                em_count += 1

            # recover
            if golden_flag:
                all_gold_text_idx = [f"text_{x}" for x in q['rel_paragraphs']]
                all_gold_table_idx = [
                    f"table_{x}" for x in q['rel_table_cols']]
                all_pred_text_idx = [f"text_{x}" for i, x in enumerate(
                    pred_text_idx) if i < text_top_k - len(all_gold_text_idx) and not (x in all_gold_text_idx)]
                all_pred_table_idx = [f"table_{x}" for i, x in enumerate(
                    pred_table_idx) if i < table_top_k - len(all_gold_table_idx) and not (x in all_gold_table_idx)]
                all_pred_idx = all_gold_text_idx + all_gold_table_idx + \
                    all_pred_text_idx + all_pred_table_idx
                random.shuffle(all_pred_idx)

            q['rel_paragraphs'] = [
                x.strip("text_") for x in all_pred_idx if x.startswith("text_")]
            q['rel_table_cols'] = [
                int(x.strip("table_")) for x in all_pred_idx if x.startswith("table_")]
            if not (0 in q['rel_table_cols']):
                q['rel_table_cols'].append(0)
            total_count += 1

    print(f"em : {em_count / total_count}")
    print(f"text em : {text_em_count / total_count}")
    print(f"table em : {table_em_count / total_count}")
    return origin_data


def get_joined_result(
    origin_data: List[Dict[str, Any]],
    top_k: int,
    golden_flag: bool
):
    em_count = 0
    total_count = 0
    for d in tqdm(origin_data):
        for q in d['questions']:
            # filter
            data[total_count]['ctxs'] = sorted(data[total_count]['ctxs'], key=(
                lambda x: x['score']), reverse=True)[:top_k]
            pred_idx = [x['idx'] for x in data[total_count]['ctxs']]
            for i, x in enumerate(pred_idx):
                if not x.startswith('text'):
                    continue
                idx = int(x.strip('text_'))
                pred_idx[i] = f"text_{idx + 1}"

            # evaluate & recover
            if 'rel_paragraphs' in q and 'rel_table_cols' in q:
                all_gold_idx = [f"text_{x}" for x in q['rel_paragraphs']
                                ] + [f"table_{x}" for x in q['rel_table_cols']]
                if len([x for x in all_gold_idx if x in pred_idx]) == len(all_gold_idx):
                    em_count += 1
                if golden_flag:
                    all_pred_idx = [x for i, x in enumerate(
                        pred_idx) if i < top_k - len(all_gold_idx) and not (x in all_gold_idx)]
                    all_pred_idx = all_gold_idx + all_pred_idx
                    random.shuffle(all_pred_idx)

            q['rel_paragraphs'] = [
                x.strip("text_") for x in all_pred_idx if x.startswith("text_")]
            q['rel_table_cols'] = [
                int(x.strip("table_")) for x in all_pred_idx if x.startswith("table_")]
            if not (0 in q['rel_table_cols']):
                q['rel_table_cols'].append(0)
            total_count += 1

    print(f"em : {em_count / total_count}")
    return origin_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--origin_data_path", "-odp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--table_top_k", type=int)
    parser.add_argument("--text_top_k", type=int)
    parser.add_argument("--use_gold", "-ug", action="store_true")
    parser.add_argument("--separated", "-s", action="store_true")
    args = parser.parse_args()
    random.seed(42)

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.origin_data_path, 'r', encoding='utf-8') as f:
        origin_data = json.load(f)
    golden_flag = (args.use_gold or "train" in args.data_path) and not (
        'test' in args.data_path)

    if args.separated:
        origin_data = get_separated_result(
            origin_data, args.text_top_k, args.table_top_k, golden_flag)
    else:
        origin_data = get_joined_result(
            origin_data, args.text_top_k + args.table_top_k, golden_flag)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(origin_data, f, ensure_ascii=False, indent=4)
