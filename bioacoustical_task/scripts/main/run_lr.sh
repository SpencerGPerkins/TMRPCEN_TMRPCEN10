#!/bin/bash
for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 1 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 1  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 1  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 1  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 1  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 1  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE


# BLOCK 2 -------------------------------
for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 2 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 2  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 2  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 2  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 2  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 2  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE

for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 3 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 3  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 3  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 3  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 3  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 3  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE

for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 4 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 4  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 4  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 4  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 4  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 4  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE

for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 5 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 5  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 5  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 5  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 5  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 5  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE


for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 6 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 6  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 6  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 6  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 6  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 6  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE



for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 7 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 7  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 7  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 7  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 7  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 7  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE



for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 8 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 8  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 8  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 8  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 8  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 8  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE



for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 9 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 9  --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 9  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 9  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 9  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 9  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE



for value in {0..4}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 10 --epoch $value --lr 0.0001 --experiment exv8
done

echo Epochs 0-4 DONE

for value in {6..9}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 10 --epoch $value --lr 0.00005 --experiment exv8
done

echo Epochs 5-9 DONE

for value in {10..14}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 10  --epoch $value --lr 0.000025 --experiment exv8
done

echo Epochs 10-14 DONE

for value in {15..19}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 10  --epoch $value --lr 0.00001 --experiment exv8
done

echo Epochs 15-19 DONE

for value in {20..24}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 10  --epoch $value --lr  0.00001 --experiment exv8
done

echo Epochs 20-24 DONE

for value in {25..29}
do
  python main_fold_epoch_lr.py --cnn birdhouse_l3_multi --frontend tmrpcen10_gpu_opt_dog --fold 10  --epoch $value --lr 0.00001 --experiment exv8
done

echo ALL DONE
