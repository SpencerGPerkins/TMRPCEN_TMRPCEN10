#!/bin/bash

for value in {1..10}
do
  python main_nofold_log.py --cnn urbanhouse_l3 --fold $value  --experiment exv2
done

echo DONE
