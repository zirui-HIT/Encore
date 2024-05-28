import sys
import json
import argparse
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

sys.path.append('.')


if __name__ == '__main__':
    from generate.utils import variable_replace
    from generate.bart.dataset import Tabular

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    parser.add_argument("--table_pointer", "-tp", action="store_true")
    parser.add_argument("--additional_process", "-ap",
                        nargs='+', default=['V', 'D', 'A'])
    parser.add_argument("--only_number", "-on", action="store_true")
    args = parser.parse_args()
    args.additional_process.extend(['C', 'F', 'S'])
    args.additional_process = [f"<{x}>" for x in args.additional_process]

    with open(args.data_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Tuple[str, str]] = []
    for d in tqdm(data):
        paragraphs: Dict[str, str] = {
            p['order']: p['text'] for p in d['paragraphs']}
        for q in d['questions']:
            # evidence info
            table = Tabular([c for i, c in enumerate(d['table']['table'])
                            if i in q['rel_table_cols']], args.table_pointer)
            table_info = table.linear("column")
            text_info = ' | '.join([paragraphs[int(x)].replace(
                '\n', '. ') for x in q['rel_paragraphs'][:6]])

            question = q['question']
            source = f"<Q> {question} | <T> {table_info} | <P> {text_info}"
            if "answer" in q:
                derivation: str = q['derivation']
                variables: List[str] = [table.index(
                    x) for x in q['variables']] if 'variables' in q else []

                answers: List[str] = q['answer'] if isinstance(
                    q['answer'], List) else [str(q['answer'])]
                answers = [x for x in answers if x]
                if q['answer_type'] == 'arithmetic':
                    if args.table_pointer:
                        answers[0] = variable_replace(derivation, variables)
                    else:
                        answers[0] = q['origin_derivation']
                        derivation = ""
                        variables = []
                elif q['answer_type'] == 'count':
                    variables = derivation.split('##')
                    derivation = ""
                else:
                    derivation = ""

                target_map: Dict[str, str] = {
                    "<C>": q['answer_type'],
                    "<F>": ' | '.join(q['answer_from'].split('-')),
                    "<V>": ' | '.join(variables),
                    "<D>": derivation,
                    "<A>": ' | '.join(answers),
                    "<S>": q['scale']
                }
                target = ' | '.join(
                    [f"{k} {v}" for k, v in target_map.items() if k in args.additional_process])
            else:
                target = "[PAD]"

            if 'train' in args.data_path and args.only_number and q['answer_type'] != 'arithmetic':
                continue
            result.append((source, target))

    with open(f"{args.output_path}.src", "w", encoding='utf-8') as f:
        f.write("\n".join([r[0] for r in result]))
    with open(f"{args.output_path}.tgt", "w", encoding='utf-8') as f:
        f.write("\n".join([r[1] for r in result]))

    print(f"preprocess {args.data_path} done")
