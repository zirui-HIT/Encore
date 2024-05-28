import sys
import json
import argparse
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

sys.path.append('.')


if __name__ == '__main__':
    from generate.bart.dataset import Tabular

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--table_pointer", "-tp", action="store_true")
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Tuple[str, str]] = []
    for d in tqdm(data):
        paragraphs: Dict[str, str] = {
            p['order']: p['text'] for p in d['paragraphs']}
        table = Tabular(d['table']['table'])
        table_info = table.linear('column')
        for q in d['questions']:
            try:
                text_info = ' | '.join([paragraphs[int(x)].replace(
                    '\n', '. ') for x in q['rel_paragraphs'][:6]])
                question = q['question']
                source = f"<Q> {question} | <T> {table_info} | <P> {text_info}"
                if "answer" in q:
                    if args.table_pointer:
                        if q['derivation']:
                            target = " | ".join(
                                [f"<D> {' | '.join(q['derivation'])}", f"<V> {' | '.join(q['variables'])}", f"<A> {' | '.join(q['answer'])}"])
                        else:
                            target = " | ".join(
                                [f"<D> {' | '.join(q['derivation'])}", f"<V> {q['origin_answer']}", f"<A> {q['origin_answer']}"])
                    else:
                        target = f"<A> {q['origin_answer'].replace('), ', ') | ')}"
                else:
                    target = "[PAD]"
                result.append((source, target))
            except:
                continue

    with open(f"{args.output_path}.src", "w", encoding='utf-8') as f:
        f.write("\n".join([r[0] for r in result]))
    with open(f"{args.output_path}.tgt", "w", encoding='utf-8') as f:
        f.write("\n".join([r[1] for r in result]))

    print(f"preprocess {args.data_path} done")
