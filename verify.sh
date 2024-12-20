#!/bin/bash
cd /root/granite-tsfm/
for i in {0..29}
do
    echo "Iteration $i"
    /home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/predict.py --checkpoint_epoch $i
    /home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/verify.py 
    /home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/test.py $i
    sleep 1
done
