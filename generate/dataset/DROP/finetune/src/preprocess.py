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
    parser.add_argument("--only_number", "-on", action="store_true")
    args = parser.parse_args()

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Tuple[str, str]] = []
    for d in tqdm(data):
        paragraphs: Dict[str, str] = {
            p['order']: p['text'] for p in d['paragraphs']}
        for q in d['questions']:
            text_info = ' | '.join([paragraphs[int(x)].replace(
                '\n', '. ') for x in q['rel_paragraphs'][:6]])
            question = q['question']
            source = f"<Q> {question} | <P> {text_info}"
            if "answer" in q:
                if args.table_pointer:
                    derivation: str = q['derivation']
                    variables = [str(x) for x in q['variables']]
                else:
                    derivation = ""
                    variables = []

                answers: List[str] = q['answer'] if isinstance(
                    q['answer'], List) else [str(q['answer'])]
                if q['answer_type'] == 'number':
                    if '+' in q['derivation'] or '-' in q['derivation']:
                        answer: str = q['derivation']
                        for i in range(len(q['variables']) - 1, -1, -1):
                            answer = answer.replace(f"x{i}", str(q['variables'][i]))
                        answers = [answer]
                    else:
                        q['answer_type'] = 'spans'

                target = " | ".join([f"<C> {q['answer_type']}",
                                     f"<V> {' | '.join(variables)}",
                                     f"<D> {derivation}",
                                     f"<A> {' | '.join(answers)}"])
            else:
                target = "[PAD]"

            if 'train' in args.data_path and args.only_number and q['answer_type'] != 'number':
                continue
            result.append((source, target))

    with open(f"{args.output_path}.src", "w", encoding='utf-8') as f:
        f.write("\n".join([r[0] for r in result]))
    with open(f"{args.output_path}.tgt", "w", encoding='utf-8') as f:
        f.write("\n".join([r[1] for r in result]))

    print(f"preprocess {args.data_path} done")
