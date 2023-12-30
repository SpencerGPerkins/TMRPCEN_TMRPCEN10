#!/bin/bash

for value in {1..10}
do
  python main_nofold.py --cnn urbanhouse_l3_multi --frontend tmrpcen_gpu --fold $value  --experiment exv3
done

echo DONE

for value in {1..10}
do
  python main_nofold.py --cnn urbanhouse_l3_multi --frontend tmrpcen10_gpu --fold $value  --experiment exv3
done

echo DONE

for value in {1..10}
do
  python main_nofold.py --cnn urbanhouse_l3_multi --frontend mrpcen --fold $value  --experiment exv3
done

echo DONE

for value in {1..10}
do
  python main_nofold.py --cnn urbanhouse_l3 --frontend pcen --fold $value  --experiment exv3
done

echo DONE
