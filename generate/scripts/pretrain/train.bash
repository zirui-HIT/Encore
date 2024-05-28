DATASET=TAT-QA
python3 ./generate/train.py \
        --dataset_path ./generate/dataset/${DATASET}/pretrain/bin \
        --exp_name None \
        --models_path ./generate/checkpoint/${DATASET}/pretrain \
        --max_tokens 8192 \
        --bart_model_path ./generate/checkpoint/PLM/BART/model.pt
