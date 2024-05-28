DATASET=TAT-QA
FILE_PATH=origin
if [ $DATASET = 'DROP' ]; then
       FILE_PATH=data
elif [ $DATASET = 'Unified' ]; then
       FILE_PATH=retrieve
fi

for PART in valid; do
       FILE_PART=$PART
       if [ $PART = 'valid' ]; then
              FILE_PART=dev
       fi
       for PLM_METHOD in BART.V_D_A; do
              MODEL_PATH=${PLM_METHOD}
              for MODEL_ID in _best; do
                     fairseq-generate \
                            --path ./generate/checkpoint/${DATASET}/finetune/${PLM_METHOD}/checkpoint${MODEL_ID}.pt \
                            ./generate/dataset/${DATASET}/finetune/bin \
                            --gen-subset ${PART} \
                            --nbest 1 \
                            --max-tokens 8192 \
                            --source-lang src \
                            --target-lang tgt \
                            --results-path ./generate/checkpoint/${DATASET}/finetune/${PLM_METHOD} \
                            --beam 5 \
                            --bpe gpt2 \
                            --remove-bpe \
                            --skip-invalid-size-inputs-valid-test
                     python3 ./generate/dataset/${DATASET}/finetune/src/evaluate.py \
                            --data_path ./generate/checkpoint/${DATASET}/finetune/${PLM_METHOD}/generate-${PART}.txt \
                            --origin_data_path ./dataset/${DATASET}/${FILE_PATH}/${FILE_PART}.json \
                            --output_path ./generate/vote/dataset/${DATASET}/${FILE_PART}.${PLM_METHOD}.json
              done
              printf "#################################################################\n"
       done
done
