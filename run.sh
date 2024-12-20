#!/bin/bash

cd /root/granite-tsfm/

# 默认情况下运行所有三个脚本
run_predict=true

# 检查命令行参数是否包含 --verify
for arg in "$@"; do
  if [ "$arg" == "--verify" ]; then
    run_predict=false
    break
  fi
done

# 运行 download_copy.py
/home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/download_copy.py

# 根据条件运行 predict.py
if [ "$run_predict" = true ]; then
  /home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/predict.py
fi

# 运行 verify.py
/home/huyaoqi/anaconda3/bin/python /root/granite-tsfm/verify.py
