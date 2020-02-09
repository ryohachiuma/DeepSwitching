import argparse
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--surgery-id', type=str, default='surgery_01')
parser.add_argument('--scale-size', type=int, default=224)
parser.add_argument('--save-raw', action='store_true', default=False)
parser.add_argument('--skip-prev', action='store_true', default=False)
parser.add_argument('--camera-id', type=int, default=None)

args = parser.parse_args()

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
FRAME_NUM = 54171
CAMERA_NUM = 5

raw_folder = './datasets/raw_video'
npy_folder = './datasets/frames'
os.makedirs(npy_folder, exist_ok=True)

files = glob.glob(os.path.join(raw_folder, args.surgery_id, '*.mp4'))
files.sort()
print(files)

frame_dir = os.path.join(npy_folder, args.surgery_id)
captures = []
for file in files:
    video = cv2.VideoCapture(file)
    captures.append(video)

for i in tqdm(range(FRAME_NUM)):
    imgs = []
    out_file = os.path.join(frame_dir, '%06d.npz' % (i))
    if args.skip_prev and os.path.isfile(out_file):
        continue
    for cap in captures:
        _, frame = cap.read()
        cv2.resize(frame, dsize=(args.scale_size, args.scale_size))
        frame = frame.astype(np.float32) // 255.0
        frame = frame[:,:,::-1]
        frame = (frame - IMG_MEAN) / IMG_STD
        imgs.append(frame)
    imgs = np.asarray(imgs) # Cam, H, W, Channel
    np.savez_compressed(out_file, imgs=imgs)
        

for cap in captures:
    cap.release()