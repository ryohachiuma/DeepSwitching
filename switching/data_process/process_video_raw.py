import argparse
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--surgery-id', type=str, default='surgery_01')
parser.add_argument('--scale-size', type=int, default=224)
parser.add_argument('--skip-prev', action='store_true', default=False)

args = parser.parse_args()

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

raw_folder = './datasets/raw_video'
npy_folder = './datasets/frames'
img_folder = './datasets/raw_frame'


files = glob.glob(os.path.join(raw_folder, args.surgery_id, '*.mp4'))
files.sort()
print(files)

frame_dir = os.path.join(npy_folder, args.surgery_id)
frame_dir_img = os.path.join(img_folder, args.surgery_id)
os.makedirs(frame_dir, exist_ok=True)
os.makedirs(frame_dir_img, exist_ok=True)
captures = []
for file in files:
    video = cv2.VideoCapture(file)
    captures.append(video)

frame_num = int(captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
for i in tqdm(range(frame_num)):
    imgs = []
    imgs_raw = []
    out_file = os.path.join(frame_dir, '%06d.npy' % (i))
    out_file_raw = os.path.join(frame_dir_img, '%06d.jpg' % (i))
    if args.skip_prev and os.path.isfile(out_file):
        continue
    for cap in captures:
        _, frame = cap.read()
        frame = cv2.resize(frame, dsize=(args.scale_size, args.scale_size))
        imgs_raw.append(frame)

        frame = frame.astype(np.float32) // 255.0
        frame = frame[:,:,::-1]
        frame = (frame - IMG_MEAN) / IMG_STD
        imgs.append(frame)
    cv2.imwrite(out_file_raw, cv2.hconcat(imgs_raw))
    imgs = np.asarray(imgs) # Cam, H, W, Channel
    np.save(out_file, imgs)
        

for cap in captures:
    cap.release()