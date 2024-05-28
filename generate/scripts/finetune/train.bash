DATASET=MathQA
PLM_METHOD=BART
MODE=BART.pointer
if [ $PLM_METHOD = 'BART' ];
then
        PLM_PATH=./generate/checkpoint/BART-large/model.pt
elif [ $PLM_METHOD = 'PT' ];
then
        PLM_PATH=./generate/checkpoint/${DATASET}/pretrain/None/checkpoint1.pt
fi

python3 ./generate/train.py \
        --dataset_path ./generate/dataset/${DATASET}/finetune/bin \
        --exp_name ${MODE} \
        --models_path ./generate/checkpoint/${DATASET}/finetune \
        --max_tokens 8192 \
        --bart_model_path ${PLM_PATH} 
