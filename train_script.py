import os
import numpy as np
import itertools
import subprocess
models = ['model_focal', 'baseline_seq', 'baseline_spac', 'baseline']
setting = range(5)

args = itertools.product(models, setting)
args = list(args)


for i in range(len(args)):
    gpu_id = i % 5
    if gpu_id == 4:
        gpu_id = 7
    cmd = 'python switching/train.py --cfg ' + args[i][0] + ' --setting ' + str(args[i][1]) + ' --gpu-index ' + str(gpu_id)
    if gpu_id == 7:
        cmd += ' --max-iter 4500'
    else:
        cmd += ' --max-iter 4000'
    print(cmd)
    if gpu_id != 7:
        subprocess.Popen(cmd, shell=True)
    else:
        os.system(cmd)

