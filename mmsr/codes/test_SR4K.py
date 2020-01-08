'''
Test 4k datasets
'''

import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


def main():
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    stage = 1  # 1 or 2, use two stage strategy for REDS dataset.
    flip_test = False
    #### model
    data_mode = 'sharp'
    if stage == 1:
        model_path = '../experiments/001_EDVRwoTSA_scratch_lr4e-4_600k_SR4K_LrCAR4S_64_20_5/models/600000_G.pth'
    else:
        model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'

    N_in = 5  # use N_in images to restore one HR image

    predeblur, HR_in = False, False
    back_RBs = 20
    if stage == 2:
        HR_in = True
        back_RBs = 20
    model = EDVR_arch.EDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in, w_TSA=True)

    #### dataset
    if stage == 1:
        test_dataset_folder = '/home/mcc/4khdr/image/540p_test'
    else:
        test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
        print('You should modify the test_dataset_folder path for stage 2')

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'sharp':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = True

    save_folder = '../results/{}'.format(data_mode)
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []

    subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    # for each subfolder
    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)

        avg_psnr, avg_psnr_border, avg_psnr_center, N_border, N_center = 0, 0, 0, 0, 0

        # process each image
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = imgs_LQ.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 1])

            logger.info('{:3d} - {:25}'.format(img_idx + 1, img_name))


if __name__ == '__main__':
    main()
