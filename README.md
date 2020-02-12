# DeepSwitching
Deep Learning model to switch cameras 

### Dataset structure
```
datasets/
  frames/
    surgery_01/
      000000.npy
      ...
    surgery_02/
  labels/
    surgery_01.csv
    surgery_02.csv
  meta/
    meta_file.yml
  raw_frame/
    surgery_01/
      000000.jpg
      ...
    surgery_02/
  raw_video/
    surgery_01/
      0.mp4
      ...
      4.mp4
```

### Data preparation
run the script to generate npy and jpg files (for visualization) of surgery (surgery_01) before training the model.
```
python switching/data_process/process_video_raw.py --surgery_id surgery_01
```

### Training the model
To train the model with configuration file (model_01) run the script file below.
```
python switching/train.py --cfg model_01 
```
TO resume the training from some iteration (iteration=500), run the script file below.
```
python switching/train.py --cfg model_01 --iter 500
```

### Results
After training the model (with cfg), results are saved with the following folder structure.
```
results/
  model_01/
    log/
    models/
    results/
    tb/
```

The testing is done like this.
```
python swtiching/train.py --cfg model_01 --mode test --data test --iter 500
```
The results are saved under the folder  results/model_01/results

