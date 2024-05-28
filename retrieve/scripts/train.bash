DATASET=FinQA
python3 ./retrieve/train.py \
        -sp ./retrieve/checkpoint/${DATASET} \
        -dp ./retrieve/dataset/${DATASET}/input
