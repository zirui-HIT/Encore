DATASET=TAT-QA
for PART in train dev test; do
        python3 ./generate/dataset/${DATASET}/finetune/src/preprocess.py \
                --data_path ./dataset/${DATASET}/retrieve/${PART}.json \
                --output_path ./generate/dataset/${DATASET}/finetune/${PART}
        python3 ./generate/multiprocessing_bpe_encoder.py \
                --encoder-json ./generate/checkpoint/BART-large/encoder.json \
                --vocab-bpe ./generate/checkpoint/BART-large/vocab.bpe \
                --inputs ./generate/dataset/${DATASET}/finetune/${PART}.src \
                --outputs ./generate/dataset/${DATASET}/finetune/${PART}.bpe.src \
                --workers 1 \
                --keep-empty
        python3 ./generate/multiprocessing_bpe_encoder.py \
                --encoder-json ./generate/checkpoint/BART-large/encoder.json \
                --vocab-bpe ./generate/checkpoint/BART-large/vocab.bpe \
                --inputs ./generate/dataset/${DATASET}/finetune/${PART}.tgt \
                --outputs ./generate/dataset/${DATASET}/finetune/${PART}.bpe.tgt \
                --workers 1 \
                --keep-empty
done
fairseq-preprocess \
        --source-lang "src" \
        --target-lang "tgt" \
        --trainpref ./generate/dataset/${DATASET}/finetune/train.bpe \
        --validpref ./generate/dataset/${DATASET}/finetune/dev.bpe \
        --testpref ./generate/dataset/${DATASET}/finetune/test.bpe \
        --destdir ./generate/dataset/${DATASET}/finetune/bin \
        --workers 2 \
        --srcdict ./generate/checkpoint/BART-large/dict.txt \
        --tgtdict ./generate/checkpoint/BART-large/dict.txt
