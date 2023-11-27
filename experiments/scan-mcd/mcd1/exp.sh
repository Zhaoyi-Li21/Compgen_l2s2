home=/data2/home/zhaoyi/spansub
for i in `seq 0 5`
do
  for ratio in 0.5
  do
  python -u $home/l2s2_main.py \
    --dataset scan \
    --seed $i \
    --scan_data_dir $home/data/SCAN \
    --scan_split mcd_d2_split \
    --scan_file mcd1 \
    --use_dev True\
    --n_epochs 150 \
    --n_enc 512 \
    --sched_factor 0.5 \
    --dropout 0.5 \
    --lr 0.001 \
    --notest_curve \
    --TEST \
    --gpu 0\
    --n_batch 128\
    --lstm_arch andreas\
    --task scan\
    --align $home/preprocess/scan/alignments/mcd1.txt\
    --fix_num_span 12\
    --warmup 80\
    > eval.$i.out 2> eval.$i.err
  done
done