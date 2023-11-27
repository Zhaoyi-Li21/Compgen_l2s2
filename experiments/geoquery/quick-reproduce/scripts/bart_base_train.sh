HOME=/data2/home/zhaoyi # e.g. change it to your own home path
TASK=$HOME/l2s2/augmentation/geoquery/comp
PARAM='-aug-base-share-ebd'

#TOTAL_NUM_UPDATES=3000
TOTAL_NUM_EPOCHS=8
#20000  
WARMUP_UPDATES=500
#500     
LR=1e-5
#3e-05
MAX_TOKENS=1024
#2048
UPDATE_FREQ=1
#32
BART_PATH=$HOME/fairseq/download_models/bart.base/model.pt # download fairseq

CUDA_VISIBLE_DEVICES=9 fairseq-train "${TASK}/bin-aug-base" \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang src --target-lang tgt \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_base \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr $LR --max-epoch $TOTAL_NUM_EPOCHS \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --save-dir "$TASK/bart-checkpoints$PARAM" \
    --save-interval-updates 500 ; 