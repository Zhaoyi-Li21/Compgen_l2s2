HOME=/data2/home/zhaoyi # e.g. change it to your own home path
TASK=$HOME/l2s2/augmentation/geoquery/comp
PARAM='-aug-base-share-ebd'

for SPLIT in train val test
do
  for LANG in src tgt
  do
    python $HOME/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
    --encoder-json $HOME/fairseq/download_models/bart.base/encoder.json \
    --vocab-bpe $HOME/fairseq/download_models/bart.base/vocab.bpe \
    --inputs "$TASK/$LANG.$SPLIT" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60;
  done
done

