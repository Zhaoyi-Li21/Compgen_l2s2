HOME=/data2/home/zhaoyi # change it to your path
TASK=$HOME/l2s2/experiments/geoquery/quick-reproduce

fairseq-preprocess \
  --source-lang "src" \
  --target-lang "tgt" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}/bin-aug-base/" \
  --workers 60 \
  --srcdict $HOME/fairseq/download_models/bart.base/dict.txt \
  --tgtdict $HOME/fairseq/download_models/bart.base/dict.txt; 