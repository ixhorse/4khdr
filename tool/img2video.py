import os
import sys
import glob
import shutil
import random
import traceback
import numpy as np
import cv2
from multiprocessing import Pool

sr4k_img_dir = '/home/mcc/working/4khdr/mmsr/results/sharp'
data_root = '/home/mcc/4khdr'
hr_video_dir = data_root + '/SDR_4K_Part1'
lr_video_dir = data_root + '/SDR_540p_test'
video_output = data_root + '/SDR_4k_test'

class VideoCombiner(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

        if not os.path.exists(self.img_dir):
            print(Fore.RED + '=> Error: ' + '{} not exist.'.format(self.img_dir))
            exit(0)

        self._get_video_shape()

    def _get_video_shape(self):
        self.all_images = sorted([os.path.join(self.img_dir, i) for i in os.listdir(self.img_dir)])
        sample_img = np.random.choice(self.all_images)
        if os.path.exists(sample_img):
            img = cv2.imread(sample_img)
            self.video_shape = img.shape
        else:
            print(Fore.RED + '=> Error: ' + '{} not found or open failed, try again.'.format(sample_img))
            exit(0)

    def combine(self, target_file='combined.mp4'):
        size = (self.video_shape[1], self.video_shape[0])
        video_writer = cv2.VideoWriter(target_file, cv2.VideoWriter_fourcc(*'XVID'), 24, size)
        for img in self.all_images:
            img = cv2.imread(img, cv2.COLOR_BGR2RGB)
            video_writer.write(img)
        video_writer.release()

def convert_worker(frame_path, video_output):
    vb = VideoCombiner(frame_path)
    vb.combine(os.path.join(video_output, os.path.basename(frame_path) + '.mp4'))

if __name__ == '__main__':
    if os.path.exists(video_output):
        shutil.rmtree(video_output)
    os.mkdir(video_output)

    test_list = [x[:-4] for x in os.listdir(lr_video_dir)]

    n_thread = 6

    pool = Pool(n_thread)
    for name in test_list:
        pool.apply_async(convert_worker, args=(os.path.join(sr4k_img_dir, name), video_output))
    pool.close()
    pool.join()
    print('Finish generating {} 4k videos.'.format(len(test_list)))

    # print(cv2.getBuildInformation())
    # video_path = glob.glob(hr_video_dir + '/*.mp4')[0]
    # videoCapture = cv2.VideoCapture(video_path)
    # fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # print(fps)
    # ex = int(videoCapture.get(cv2.CAP_PROP_FOURCC))
    # # ex = cv2.VideoWriter_fourcc(*'H265')
    # print(chr(ex&0xFF) + chr((ex>>8)&0xFF) + chr((ex>>16)&0xFF) + chr((ex>>24)&0xFF))
