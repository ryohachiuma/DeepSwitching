import os
import numpy as np
import itertools
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', nargs='*', default='model_01')
parser.add_argument('--max-iter', type=int, default=4000)
parser.add_argument('--surgery-out', action='store_true')
args = parser.parse_args()
models = args.cfg

if args.surgery_out:
    set_num = 14
else:
    set_num = 5
setting = range(set_num)

args = itertools.product(models, setting)
args = list(args)


for i in range(len(args)):
    gpu_id = i % set_num
    if gpu_id == 4:
        gpu_id = 7
    cmd = 'python switching/train.py --cfg ' + args[i][0] + ' --setting ' + str(args[i][1]) + ' --gpu-index ' + str(gpu_id)
    if gpu_id == 7:
        cmd += ' --max-iter ' + str(args.max_iter + 500)
    else:
        cmd += ' --max-iter ' + str(args.max_iter)
    print(cmd)
    if gpu_id != 7:
        subprocess.Popen(cmd, shell=True)
    else:
        os.system(cmd)

