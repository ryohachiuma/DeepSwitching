import argparse
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def preprocess(frame):
    frame = frame.astype(np.float32) // 255.0
    frame = frame[:,:,::-1]
    frame = (frame - IMG_MEAN) / IMG_STD
    return frame

parser = argparse.ArgumentParser()
parser.add_argument('--surgery-id', type=str, default='surgery_01')
parser.add_argument('--scale-size', type=int, default=224)
parser.add_argument('--skip-prev', action='store_true', default=False)

args = parser.parse_args()



raw_folder = './datasets/raw_video'
npy_folder = './datasets/frames'
img_folder = './datasets/raw_frame'


files = glob.glob(os.path.join(raw_folder, args.surgery_id, '*.mp4'))
files.sort()
print(files)

splitted = False
if len(files) > 1:
    splitted = True

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
    if splitted:
        for cap in captures:
            _, frame = cap.read()
            print(frame.shape)
            frame = cv2.resize(frame, dsize=(args.scale_size, args.scale_size))
            imgs_raw.append(frame)

            imgs.append(preprocess(frame))
    else:
        cap = captures[0]
        _, frame = cap.read()
        w = frame.shape[1]
        h = frame.shape[0]
        frame_0 = frame[:int(h / 2), :int(w / 3), :]
        frame_0 = cv2.resize(frame_0, dsize=(args.scale_size, args.scale_size))
        imgs_raw.append(frame_0)
        imgs.append(preprocess(frame_0))

        frame_1 = frame[:int(h / 2), int(w / 3): int(w * 2 / 3), :]
        frame_1 = cv2.resize(frame_1, dsize=(args.scale_size, args.scale_size))
        imgs_raw.append(frame_1)
        imgs.append(preprocess(frame_1))

        frame_2 = frame[:int(h / 2), int(w * 2 / 3):, :]
        frame_2 = cv2.resize(frame_2, dsize=(args.scale_size, args.scale_size))
        imgs_raw.append(frame_2)
        imgs.append(preprocess(frame_2))

        frame_3 = frame[int(h / 2):, :int(w / 3), :]
        frame_3 = cv2.resize(frame_3, dsize=(args.scale_size, args.scale_size))
        imgs_raw.append(frame_3)
        imgs.append(preprocess(frame_3))

        frame_4 = frame[int(h / 2):, int(w / 3):int(w * 2 / 3), :]
        frame_4 = cv2.resize(frame_4, dsize=(args.scale_size, args.scale_size))
        imgs_raw.append(frame_4)
        imgs.append(preprocess(frame_4))


    cv2.imwrite(out_file_raw, cv2.hconcat(imgs_raw))
    imgs = np.asarray(imgs) # Cam, H, W, Channel
    np.save(out_file, imgs)
        

for cap in captures:
    cap.release()