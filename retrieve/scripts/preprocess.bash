DATASET=FinQA
for PART in train dev test
do
        python3 ./retrieve/generate_dataset.py \
                -dp ./dataset/${DATASET}/origin/${PART}.json \
                -op ./retrieve/dataset/${DATASET}/input/${PART}.json
done
