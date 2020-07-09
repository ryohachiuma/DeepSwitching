import os
import numpy as np
import itertools
import subprocess

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', nargs='*', default='model_01')
parser.add_argument('--min-iter', type=int, default=3000)
parser.add_argument('--max-iter', type=int, default=4000)
parser.add_argument('--surgery-out', action='store_true')

models = args.cfg

if args.surgery_out:
    set_num = 14
else:
    set_num = 5

models = args.cfg

_range = np.arange(args.min_iter, args.max_iter, 500)

setting = range(5)

args = itertools.product(models, _range.tolist(), setting)
args = list(args)


for i in range(len(args)):
    gpu_id = i % 7
    if gpu_id == 4:
        gpu_id = 7
    cmd = 'python switching/train.py --mode test --data test --cfg ' + args[i][0] + ' --iter ' + str(args[i][1]) + ' --setting ' + str(args[i][2]) + ' --gpu-index ' + str(gpu_id)
    print(cmd)
    os.system(cmd)
