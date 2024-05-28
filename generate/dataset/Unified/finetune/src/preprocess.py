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
    parser.add_argument("--additional_process", "-ap", type=str)
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Tuple[str, str]] = []
    for d in tqdm(data):
        paragraphs: Dict[str, str] = {
            p['order']: p['text'] for p in d['paragraphs']}
        if 'table' in d['table']:
            table = Tabular(d['table']['table'], args.table_pointer)
            table_info = table.linear("column")
        else:
            table = None
            table_info = ''

        for q in d['questions']:
            if 'rel_paragraphs' in q:
                text_info = ' | '.join([paragraphs[int(x)].replace(
                    '\n', '. ') for x in q['rel_paragraphs'][:6]])
            else:
                text_info = ''
            question = q['question']
            source = f"<Q> {question} | <T> {table_info} | <P> {text_info}"

            if "answer" in q:
                if args.table_pointer:
                    derivation: str = ' | '.join(q['derivation'])
                    variables = [table.index(
                        x) for x in q['variables']] if table else q['variables']

                    answer = derivation
                    for i in range(len(variables) - 1, -1, -1):
                        answer = answer.replace(f"x{i}", variables[i])
                else:
                    derivation = ''
                    variables = []
                    answer = ' | '.join(q['answer'])

                target = " | ".join([f"<V> {' | '.join(variables)}",
                                     f"<D> {derivation}",
                                     f"<A> {answer}"])
            else:
                target = "[PAD]"

            result.append((source, target))

    with open(f"{args.output_path}.src", "w", encoding='utf-8') as f:
        f.write("\n".join([r[0] for r in result]))
    with open(f"{args.output_path}.tgt", "w", encoding='utf-8') as f:
        f.write("\n".join([r[1] for r in result]))

    print(f"preprocess {args.data_path} done")
