import sys
import json
import argparse
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

sys.path.append('.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--table_pointer", "-tp", action="store_true")
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Tuple[str, str]] = []
    for d in tqdm(data):
        for q in d['questions']:
            question = q['question']
            source = f"<Q> {question}"
            if "answer" in q:
                if args.table_pointer:
                    target = " | ".join([f"<D> {' | '.join(q['derivation'])}",
                                         f"<A> {' | '.join(q['answer'])}"])
                else:
                    target = f"<A> {' | '.join(q['answer'])}"
            else:
                target = "[PAD]"
            result.append((source, target))

    with open(f"{args.output_path}.src", "w", encoding='utf-8') as f:
        f.write("\n".join([r[0] for r in result]))
    with open(f"{args.output_path}.tgt", "w", encoding='utf-8') as f:
        f.write("\n".join([r[1] for r in result]))

    print(f"preprocess {args.data_path} done")
