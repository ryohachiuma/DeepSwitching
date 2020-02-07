import argparse
import os
import glob
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--surgery-id', type=str, default='surgery_01')
parser.add_argument('--crop-size', type=int, default=1000)
parser.add_argument('--scale-size', type=int, default=224)
args = parser.parse_args()

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

raw_folder = './datasets/raw_video'
frames_folder = './datasets/frames_raw'
npy_folder = './datasets/frames'
os.makedirs(frames_folder, exist_ok=True)

files = glob.glob(os.path.join(raw_folder, args.surgery_id, '*.mp4'))
files.sort()
print(files)
camera_id = 0
for file in files:
    frame_dir = os.path.join(frames_folder, args.surgery_id, str(camera_id))
    npy_dir = os.path.join(npy_folder, args.surgery_id, str(camera_id))
    os.makedirs(frame_dir, exist_ok=True)  
    os.makedirs(npy_dir, exist_ok=True)
    video = cv2.VideoCapture(file)
    frame_id = 0
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.resize(frame, dsize=(args.scale_size, args.scale_size))
        out_file = os.path.join(frame_dir, '%06d.png' % (frame_id))
        cv2.imwrite(out_file, frame)

        out_file = os.path.join(npy_dir, '%06d.npz' % (frame_id))
        frame = frame.astype(np.float32)
        frame = (frame - IMG_MEAN) / IMG_STD

        np.savez(out_file, frame)

        frame_id+=1
    video.release()
    camera_id+=1