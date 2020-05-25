import os
import numpy as np

_range = np.arange(1000, 10000, 500)

for i in _range:

    for c in range(5):
        command = 'python switching/train.py --mode test --data test --cfg model_focal --iter ' + str(i) + ' --setting ' + str(c)  
        print(command)
        os.system(command)