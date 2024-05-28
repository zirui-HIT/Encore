MODEL_PATH=bert-base-hybrid
for DATASET in TAT-QA
do
        for PART in dev
        do
                python3 ./retrieve/infer.py \
                        --data_path ./retrieve/dataset/${DATASET}/input \
                        --save_path ./retrieve/checkpoint/TAT-QA/${MODEL_PATH}/saved_model/model.pt \
                        --log_path ./retrieve/checkpoint/TAT-QA/${MODEL_PATH}/results/log.txt \
                        --output_path ./retrieve/dataset/${DATASET}/output/${PART}.json \
                        --part ${PART}
                python3 ./retrieve/postprocess.py \
                        --data_path ./retrieve/dataset/${DATASET}/output/${PART}.json \
                        --origin_data_path ./dataset/${DATASET}/origin/${PART}.json \
                        --output_path ./dataset/${DATASET}/retrieve/${PART}.json \
                        --table_top_k 8 \
                        --text_top_k 0 \
                        --use_gold
        done
done
