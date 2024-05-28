import argparse
import subprocess


def run_command(bash_command):
    process = subprocess.Popen(bash_command.split())
    output, error = process.communicate()
    print(error)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--models_path", type=str)
    parser.add_argument("--bart_model_path", type=str)
    parser.add_argument("--max_tokens", type=int)
    args = parser.parse_args()

    max_epoch = 200
    save_interval = 10
    if 'pretrain' in args.dataset_path:
        max_epoch = 1
        save_interval = 1
    if 'DROP' in args.dataset_path:
        max_epoch = 10
        save_interval = 1

    print("START training")
    run_command("printenv")

    cmd = f"""
    fairseq-train {args.dataset_path} \
    --save-dir {args.models_path}/{args.exp_name} \
    --restore-file {args.bart_model_path} \
    --arch bart_large  \
    --criterion label_smoothed_cross_entropy  \
    --source-lang src  \
    --target-lang tgt  \
    --truncate-source  \
    --label-smoothing 0.1  \
    --max-tokens {args.max_tokens}  \
    --max-epoch {max_epoch} \
    --save-interval {save_interval}
    --update-freq 4  \
    --required-batch-size-multiple 1  \
    --dropout 0.1  \
    --attention-dropout 0.1  \
    --relu-dropout 0.0  \
    --weight-decay 0.05  \
    --optimizer adam  \
    --adam-eps 1e-08  \
    --clip-norm 0.1  \
    --lr-scheduler polynomial_decay  \
    --lr 1e-05  \
    --warmup-updates 5000  \
    --ddp-backend no_c10d  \
    --num-workers 20  \
    --reset-meters  \
    --reset-optimizer \
    --reset-dataloader \
    --share-all-embeddings \
    --layernorm-embedding \
    --share-decoder-input-output-embed  \
    --skip-invalid-size-inputs-valid-test  \
    --log-format json  \
    --log-interval 5000  \
    --validate-interval 10 \
    --patience 200 \
    --no-last-checkpoints \
    --no-save-optimizer-state \
    --report-accuracy \
    --fp16
    """

    print("RUN {}".format(cmd))
    run_command(cmd)
