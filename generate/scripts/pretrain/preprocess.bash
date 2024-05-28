DATASET=FinQA
python3 ./generate/dataset/${DATASET}/pretrain/src/preprocess.py \
        -dp ./dataset/${DATASET}/retrieve \
        -op ./generate/dataset/${DATASET}/pretrain \
        -tp
for PART in dev train test
do
        python3 ./generate/multiprocessing_bpe_encoder.py \
                --encoder-json ./generate/checkpoint/PLM/BART/encoder.json \
                --vocab-bpe ./generate/checkpoint/PLM/BART/vocab.bpe \
                --inputs ./generate/dataset/${DATASET}/pretrain/${PART}.src \
                --outputs ./generate/dataset/${DATASET}/pretrain/${PART}.bpe.src \
                --workers 1 \
                --keep-empty
        python3 ./generate/multiprocessing_bpe_encoder.py \
                --encoder-json ./generate/checkpoint/PLM/BART/encoder.json \
                --vocab-bpe ./generate/checkpoint/PLM/BART/vocab.bpe \
                --inputs ./generate/dataset/${DATASET}/pretrain/${PART}.tgt \
                --outputs ./generate/dataset/${DATASET}/pretrain/${PART}.bpe.tgt \
                --workers 1 \
                --keep-empty
done
fairseq-preprocess \
        --source-lang "src" \
        --target-lang "tgt" \
        --trainpref ./generate/dataset/${DATASET}/pretrain/train.bpe \
        --validpref ./generate/dataset/${DATASET}/pretrain/dev.bpe \
        --destdir ./generate/dataset/${DATASET}/pretrain/bin \
        --workers 2 \
        --srcdict ./generate/checkpoint/PLM/BART/dict.txt \
        --tgtdict ./generate/checkpoint/PLM/BART/dict.txt
