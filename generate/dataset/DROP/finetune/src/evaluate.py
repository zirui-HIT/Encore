import re
import sys
import json
import string
import argparse
import numpy as np
from re import RegexFlag
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, List, Union, Tuple, Set
from scipy.optimize import linear_sum_assignment


sys.path.append(".")


def _remove_articles(text: str) -> str:
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)


def _white_space_fix(text: str) -> str:
    return ' '.join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    if not _is_number(text):
        return ''.join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    return text.lower()


def _tokenize(text: str) -> List[str]:
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    parts = [_white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token)))))
             for token in _tokenize(text)]
    parts = [part for part in parts if part.strip()]
    normalized = ' '.join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(
                    pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision +
                                     recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]]) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        return tuple(["{0} {1} {2}".format(answer["date"]["day"],
                                           answer["date"]["month"],
                                           answer["date"]["year"])]), "date"
    else:
        raise ValueError(
            f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}")


def evaluate_json(annotations: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float, List[bool]]:
    """
    Takes gold annotations and predicted answers and  evaluates the predictions for each question
    in the gold annotations.  Both JSON dictionaries must have query_id keys, which are used to
    match predictions to gold annotations (note that these are somewhat deep in the JSON for the
    gold annotations, but must be top-level keys in the predicted answers).
    The ``annotations`` are assumed to have the format of the dev set in the DROP data release.
    The ``predicted_answers`` JSON must be a dictionary keyed by query id, where the value is a string
    (or list of strings) that is the answer.
    """
    instance_exact_match = []
    instance_f1 = []
    # for each type as well
    match_flag: List[bool] = []
    type_to_em: Dict[str, List[float]] = defaultdict(list)
    type_to_f1: Dict[str, List[float]] = defaultdict(list)
    for _, annotation in annotations.items():
        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            max_em_score = 0.0
            max_f1_score = 0.0
            max_type = None
            if query_id in predicted_answers:
                predicted = predicted_answers[query_id]
                candidate_answers = [qa_pair["answer"]]
                if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                    candidate_answers += qa_pair["validated_answers"]
                for answer in candidate_answers:
                    gold_answer, gold_type = answer_json_to_strings(answer)
                    em_score, f1_score = get_metrics(predicted, gold_answer)
                    if gold_answer[0].strip() != "":
                        max_em_score = max(max_em_score, em_score)
                        max_f1_score = max(max_f1_score, f1_score)
                        if max_em_score == em_score or max_f1_score == f1_score:
                            max_type = gold_type
            else:
                print("Missing prediction for question: {}".format(query_id))
                if qa_pair and qa_pair["answer"]:
                    max_type = answer_json_to_strings(qa_pair["answer"])[1]
                else:
                    max_type = "number"
                max_em_score = 0.0
                max_f1_score = 0.0
            match_flag.append(abs(1.0 - max_em_score) < 1e-8)
            instance_exact_match.append(max_em_score)
            instance_f1.append(max_f1_score)
            type_to_em[max_type].append(max_em_score)
            type_to_f1[max_type].append(max_f1_score)

    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        print("{0}: {1} ({2:.2f}%)".format(
            typ, len(type_to_em[typ]), 100. * len(type_to_em[typ])/total))
        print(
            "  Exact-match accuracy {0:.3f}".format(100. * np.mean(type_to_em[typ])))
        print("  F1 score {0:.3f}".format(100. * np.mean(type_to_f1[typ])))
    return global_em, global_f1, match_flag


def extract_structure_data(plain_text_content: str, origin_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_by_id(data):
        data.sort(key=lambda x: int(x.split('\t')[0][2:]))
        return data

    predict_outputs = sort_by_id(re.findall(
        "^D.+", plain_text_content, RegexFlag.MULTILINE))
    ground_outputs = sort_by_id(re.findall(
        "^T.+", plain_text_content, RegexFlag.MULTILINE))
    source_inputs = sort_by_id(re.findall(
        "^S.+", plain_text_content, RegexFlag.MULTILINE))

    def parse_answer(text: str) -> Dict[str, Any]:
        answers: List[str] = []
        answer_type = ""
        current_type = 'None'
        for word in text.split('|'):
            word = word.strip()
            if word.startswith('<C>'):
                answer_type = word.strip('<C>').strip()
                current_type = 'C'
            elif word.startswith('<A>'):
                answers.append(word.strip('<A>').strip())
                current_type = 'A'
            elif current_type == 'A':
                answers.append(word.strip())
        if not answers:
            answers.append('0')

        result: Dict[str, Any] = {
            "type": answer_type,
            "origin_pred": text
        }
        if answer_type == 'spans':
            result['answer'] = answers
        elif answer_type == 'number':
            try:
                result['answer'] = str(calc_derivation(answers[0])[0])
            except:
                result['answer'] = ''
        elif answer_type == 'count':
            result['type'] = 'number'
            result['answer'] = answers[0]
        elif answer_type == 'date':
            result['answer'] = {
                "day": "",
                "month": "",
                "year": ""
            }
            for s in answers[0].split('##'):
                if len(s) == 4 and s.isdigit():
                    result['answer']['year'] = s
                elif s.isdigit():
                    result['answer']['day'] = s
                else:
                    result['answer']['month'] = s
        else:
            raise Exception(f'illegal answer type : {answer_type}')

        return result

    idx = 0
    data: List[Dict[str, Any]] = []
    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        predict_id, predict_score, predict_clean = predict.split('\t')
        ground_id, ground_clean = ground.split('\t')
        source_id, source_clean = source.split('\t')

        real_pred_id = int(predict_id.split('-')[-1])
        while idx < real_pred_id:
            data.append({
                "id": f"D-{idx}",
                "pred": {},
                "gold": {},
                "source": ""
            })
            idx += 1

        try:
            pred_answer = parse_answer(predict_clean)
        except Exception as e:
            print(f"An error occurred in source: {idx}, {e}")
            pred_answer = {"origin_pred": predict_clean}
        try:
            gold_answer = parse_answer(ground_clean)
        except:
            gold_answer = {"origin_gold": ground_clean}

        data.append({
            "id": predict_id,
            "pred": pred_answer,
            "gold": gold_answer,
            "source": source_clean
        })
        idx += 1

    return data


def process_file(generate_file_path: str) -> None:
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content, origin_data)

    idx = 0
    predict_data: Dict[str, Tuple[str, ...]] = {}
    for d in origin_data.values():
        for i in range(len(d['qa_pairs'])):
            x = data[idx]['pred']
            current_answer = {
                "number": "",
                "date": {
                    "day": "",
                    "month": "",
                    "year": ""
                },
                "spans": []
            }
            current_answer[x['type']] = x['answer']
            predict_data[d["qa_pairs"][i]['query_id']
                         ] = answer_json_to_strings(current_answer)[0]
            idx += 1
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(predict_data, f, ensure_ascii=False, indent=4)

    if not ('test' in generate_file_path):
        em, f1, flags = evaluate_json(origin_data, predict_data)
        for i, d in enumerate(data):
            data[i]['correct'] = flags[i]
        print(f"em : {em} , f1 : {f1}")
        with open(args.output_path.replace('.json', '.cases.json'), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    from generate.utils import calc_derivation

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-dp", type=str,
                        default='./generate/bart/checkpoint/DROP/finetune/BART.pointer/generate-valid.txt')
    parser.add_argument("--origin_data_path", "-odp", type=str,
                        default='./dataset/DROP/data/dev.json')
    parser.add_argument("--output_path", "-op", type=str,
                        default='./generate/vote/dataset/DROP/dev.BART.pointer.json')
    parser.add_argument("--default_score", "-ds", type=float, default=None)
    args = parser.parse_args()

    if not (args.origin_data_path is None):
        with open(args.origin_data_path, 'r', encoding='utf-8') as f:
            origin_data: Dict[str, Dict[str, Any]] = json.load(f)

    process_file(args.data_path)
