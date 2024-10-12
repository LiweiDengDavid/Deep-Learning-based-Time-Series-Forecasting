for pred_Len in 48 96 336
do
for model_name in TimeMixer
do
for rate in 0.05 0.01 0.005 0.001 0.0001
do
python -u main.py \
  --model_name=$model_name \
  --train=1 \
  --exp='deep_learning'\
  --seed=1 \
  --data_name=ETTm1 \
  --seq_len=96 \
  --label_len=48 \
  --pred_len=$pred_Len\
  --d_mark=4 \
  --d_feature=7 \
  --c_out=7 \
  --features='M' \
  --d_model=510 \
  --n_heads=3\
  --d_ff=1024 \
  --dropout=0.05\
  --lr=$rate \
  --lr_d=0.05\
  --batch_size=16 \
  --patience=5 \
  --epoches=100\

done
done
done