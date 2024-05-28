import sys
import json
import argparse

from typing import List, Dict, Any


sys.path.append('.')


if __name__ == '__main__':
    from retrieve.general_utils import table_row_to_text

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Dict[str, Any]] = []
    for i, d in enumerate(data):
        for j, q in enumerate(d['questions']):
            result.append({})
            result[-1]['pre_text'] = [x['text'] for x in d['paragraphs']]
            result[-1]['post_text'] = []
            result[-1]['id'] = f"{i}_{j}"
            result[-1]['table'] = d['table']['table']
            result[-1]['table_ori'] = d['table']['table']
            result[-1]['qa'] = {
                "question": q['question'],
                "ann_text_rows": [int(x) - 1 for x in q['rel_paragraphs']] if 'rel_paragraphs' in q else [],
                "ann_table_rows": q['rel_table_cols'] if 'rel_table_cols' in q else []
            }

            gold_inds = {}
            if 'rel_paragraphs' in q:
                gold_inds.update(
                    {f"text_{int(x) - 1}": d['paragraphs'][int(x) - 1]['text'] for x in q['rel_paragraphs']})
            if 'rel_table_cols' in q:
                gold_inds.update({f"table_{i+1}": table_row_to_text(
                    d['table']['table'][0], x) for i, x in enumerate(d['table']['table'][1:])})
            result[-1]['qa']['gold_inds'] = gold_inds

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    print(f"preprocess {args.data_path} done")
