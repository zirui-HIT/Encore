import re
import json
import argparse
from re import RegexFlag
from typing import Any, Dict, List


def extract_structure_data(plain_text_content: str) -> List[Dict[str, Any]]:
    def sort_by_id(data):
        data.sort(key=lambda x: int(x.split('\t')[0][2:]))
        return data

    predict_outputs = sort_by_id(re.findall(
        "^D.+", plain_text_content, RegexFlag.MULTILINE))
    ground_outputs = sort_by_id(re.findall(
        "^T.+", plain_text_content, RegexFlag.MULTILINE))
    source_inputs = sort_by_id(re.findall(
        "^S.+", plain_text_content, RegexFlag.MULTILINE))

    data: List[Dict[str, Any]] = []
    for idx, (predict, ground, source) in enumerate(zip(predict_outputs, ground_outputs, source_inputs)):
        predict_id, predict_score, predict_clean = predict.split('\t')
        ground_id, ground_clean = ground.split('\t')
        source_id, source_clean = source.split('\t')

        data.append({
            "id": predict_id,
            "pred": predict_clean.strip(),
            "gold": ground_clean.strip(),
            "source": source_clean.strip()
        })

    return data


def evaluate(data: List[Dict[str, Any]]) -> List[bool]:
    exact_match = sum([1 if d['pred'] == d['gold'] else 0 for d in data])
    return exact_match / len(data)


def process_file(generate_file_path: str) -> None:
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content)
    print(evaluate(data))

    bad_cases: List[Dict[str, Any]] = [
        d for d in data if d['pred'] != d['gold']]
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()

    process_file(args.data_path)
