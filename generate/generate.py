import os
import json
import argparse
from typing import List, Dict, Any

from train import run_command


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", "-cp", type=str)
    parser.add_argument("--output_path", "-op", type=str)
    args = parser.parse_args()

    valid_dataset = "total" if "total" in args.output_path else "part"
    model_dataset = "total" if "total" in args.ckpt_path else "part"
    model_filename: List[str] = os.listdir(args.ckpt_path)

    for filename in model_filename:
        data_type, plm_type, time_tag = filename.split('.')

        generate_command = f"""
        fairseq-generate \
        --path  ./generate/seq2seq/checkpoint/finetune/{model_dataset}/{filename}/best_checkpoint.pt \
                ./generate/seq2seq/dataset/finetune/{data_type}/{valid_dataset}/bin \
        --gen-subset valid \
        --nbest 1 \
        --max-tokens 8192 \
        --source-lang src \
        --target-lang tgt \
        --results-path ./generate/seq2seq/checkpoint/finetune/{model_dataset}/{filename} \
        --beam 5 \
        --bpe gpt2 \
        --remove-bpe \
        --skip-invalid-size-inputs-valid-test
        """
        run_command(generate_command)

        # load default score
        default_score = None
        if 'total' in args.output_path:
            default_score = json.load(open(
                f"{args.output_path}/dev.seq2seq_{data_type}.{plm_type}.json".replace('total', 'part')))[0]['pred']['score']
        postprocess_command = f"""
        python3 ./generate/seq2seq/dataset/finetune/evaluate.py \
        -dp ./generate/seq2seq/checkpoint/finetune/{model_dataset}/{filename}/generate-valid.txt \
        -odp ./dataset/{valid_dataset}/origin/dev.json \
        -op {args.output_path}/dev.seq2seq_{data_type}.{plm_type}.json
        """
        if not(default_score is None):
            postprocess_command = f"{postprocess_command} -ds {default_score}"
        run_command(postprocess_command)

    print(f"generate {args.ckpt_path} done")
