HOME=/data2/home/zhaoyi # e.g. change it to your own home path
TASK=/$HOME/l2s2/augmentation/geoquery/comp
PARAM='-aug-base-share-ebd'

for f in $HOME/l2s2/experiments/geoquery/quick-reproduce/bart-checkpoints-aug-base-share-ebd/*
do 
    echo '------------------------------------------------------------'
    echo $f
    CUDA_VISIBLE_DEVICES=5 python $HOME/l2s2/utils/geoquery/semantic_parsing.py --model-dir $(dirname $f) --model-file $(basename $f) --src $TASK/src.test --out $TASK/pred$PARAM.test --data-dir "${TASK}/bin-aug-base/" 
    python $HOME/l2s2/utils/geoquery/bart_acc.py --pred $TASK/pred$PARAM.test --gold $TASK/tgt.test.gt
done 
