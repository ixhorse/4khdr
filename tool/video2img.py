import os
import sys
import glob
import shutil
import random
import traceback
import numpy as np
import cv2
from multiprocessing import Pool


data_root = '/home/mcc/4khdr'
# video path
gt_dirs = [os.path.join(data_root, 'SDR_4K_Part%d' % x) for x in range(1, 5)]
lr_dir = os.path.join(data_root, 'SDR_540p')
test_dir = os.path.join(data_root, 'SDR_540p_test')

# image path
image_dir = data_root + '/image'
image_4k_dir = image_dir + '/4k'
image_540p_dir = image_dir + '/540p'
image_test_dir = image_dir + '/540p_test'

def convert_worker(video_path, img_dir, split=False):
    # convert video to frames
    img_dir_base = os.path.join(img_dir, os.path.basename(video_path)[:-4])
    if split:
        for i in range(4):
            os.mkdir(img_dir_base + 'x%d' % i)
    else:
        if not os.path.exists(img_dir_base):
            os.mkdir(img_dir_base)

    videoCapture = cv2.VideoCapture(video_path)
    cnt = 0
    success, frame = videoCapture.read()
    h, w = frame.shape[:2]
    while(success):
        if split:
            for i in range(2):
                for j in range(2):
                    try:
                        cv2.imwrite(img_dir_base + 'x{}/{:04d}.png'.format(i*2+j, cnt),
                                    frame[int(i * h / 2) : int((i+1) * h / 2),
                                        int(j * w / 2) : int((j+1) * w / 2),
                                        :], 
                                    [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
                    except Exception as e:
                        traceback.print_exc()
        else:
            cv2.imwrite(img_dir_base + '/{:04d}.png'.format(cnt), frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cnt += 1
        success, frame = videoCapture.read()


if __name__ == '__main__':
    # train_all_list = os.listdir(lr_dir)
    # random.shuffle(train_all_list)
    # train_list = train_all_list[:int(len(train_all_list) * 0.9)]
    # val_list = train_all_list[int(len(train_all_list) * 0.9):]
    # test_list = os.listdir(test_dir)

    # with open(data_root + '/train.txt', 'w') as f:
    #     data = [x[:-4] + '\n' for x in train_list]
    #     f.writelines(data)
    # with open(data_root + '/val.txt', 'w') as f:
    #     data = [x[:-4] + '\n' for x in val_list]
    #     f.writelines(data)
    # with open(data_root + '/test.txt', 'w') as f:
    #     data = [x[:-4] + '\n' for x in test_list]
    #     f.writelines(data)

    # if os.path.exists(image_dir):
    #     shutil.rmtree(image_dir)
    # os.mkdir(image_dir)
    # os.mkdir(image_4k_dir)
    # os.mkdir(image_540p_dir)
    # os.mkdir(image_test_dir)

    with open('/home/mcc/4khdr/val.txt') as f:
        val_list = [x.strip() for x in f.readlines()]

    hr_video_paths = []
    for gt_dir in gt_dirs:
        hr_video_paths += glob.glob(gt_dir + '/*.mp4')
    hr_video_paths = [x for x in hr_video_paths if os.path.basename(x)[:-4] in val_list]
    lr_video_paths = [os.path.join(lr_dir, name+'.mp4') for name in val_list]
    test_video_paths = glob.glob(test_dir + '/*.mp4')
    
    n_thread = 12

    pool = Pool(n_thread)
    for path in lr_video_paths:
        pool.apply_async(convert_worker, args=(path, image_540p_dir, False))
    pool.close()
    pool.join()
    print('Finish converting {} 540p videos.'.format(len(lr_video_paths)))

    pool = Pool(n_thread)
    for path in hr_video_paths:
        pool.apply_async(convert_worker, args=(path, image_4k_dir, False))
    pool.close()
    pool.join()
    print('Finish converting {} 4k videos.'.format(len(hr_video_paths)))

    # pool = Pool(n_thread)
    # for path in test_video_paths:
    #     pool.apply_async(convert_worker, args=(path, image_test_dir, False))
    # pool.close()
    # pool.join()
    # print('Finish converting {} test videos.'.format(len(test_video_paths)))