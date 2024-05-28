DATASET=TAT-QA
fairseq-generate \
       --path ./generate/checkpoint/${DATASET}/pretrain/checkpoint1.pt \
              ./generate/dataset/${DATASET}/pretrain/bin \
       --gen-subset valid \
       --nbest 1 \
       --max-tokens 8192 \
       --source-lang src \
       --target-lang tgt \
       --results-path ./generate/checkpoint/${DATASET}/pretrain \
       --beam 5 \
       --bpe gpt2 \
       --remove-bpe \
       --skip-invalid-size-inputs-valid-test
python3 ./generate/dataset/${DATASET}/pretrain/src/evaluate.py \
       -dp ./generate/checkpoint/${DATASET}/pretrain/generate-valid.txt \
       -op ./generate/checkpoint/${DATASET}/pretrain/generate-valid.json
