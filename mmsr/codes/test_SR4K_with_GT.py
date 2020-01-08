'''
Test 4k datasets
'''

import os
import sys
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
    data_mode = 'sharp_bicubic'
    if stage == 1:
        model_path = '../experiments/001_EDVRwoTSA_scratch_lr4e-4_600k_SR4K_LrCAR4S/models/200000_G.pth'
    else:
        model_path = '../experiments/pretrained_models/EDVR_REDS_SR_Stage2.pth'

    N_in = 3  # use N_in images to restore one HR image

    predeblur, HR_in = False, False
    back_RBs = 10
    if stage == 2:
        HR_in = True
        back_RBs = 20
    model = EDVR_arch.EDVR(64, N_in, 8, 5, back_RBs, predeblur=predeblur, HR_in=HR_in, w_TSA=False)

    #### dataset
    val_txt = '/home/mcc/4khdr/val.txt'
    if stage == 1:
        test_dataset_folder = '/home/mcc/4khdr/image/540p'
        GT_dataset_folder = '/home/mcc/4khdr/image/4k'
    else:
        test_dataset_folder = '../results/REDS-EDVR_REDS_SR_L_flipx4'
        print('You should modify the test_dataset_folder path for stage 2')

    #### evaluation
    crop_border = 0
    border_frame = N_in // 2  # border frames when evaluate
    # temporal padding mode
    if data_mode == 'sharp_bicubic':
        padding = 'new_info'
    else:
        padding = 'replicate'
    save_imgs = False

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

    avg_psnr_l, avg_psnr_center_l, avg_psnr_border_l = [], [], []
    subfolder_name_l = []

    with open(val_txt, 'r') as f:
        split_name_l = []
        name_l = [x.strip() for x in f.readlines()]
        # for name in name_l:
        #     for i in range(4):
        #         split_name_l.append(name + 'x{}'.format(i))
    subfolder_l = sorted([osp.join(test_dataset_folder, name) for name in name_l])
    subfolder_GT_l = sorted([osp.join(GT_dataset_folder, name) for name in name_l])
    # for each subfolder
    for subfolder, subfolder_GT in zip(subfolder_l, subfolder_GT_l):
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        img_GT_l = []
        for img_GT_path in sorted(glob.glob(osp.join(subfolder_GT, '*'))):
            img_GT_l.append(data_util.read_img(None, img_GT_path))

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
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            # calculate PSNR
            output = output / 255.
            GT = np.copy(img_GT_l[img_idx])
            # evaluate on RGB channels
            output, GT = util.crop_border([output, GT], crop_border)
            crt_psnr = util.calculate_psnr(output * 255, GT * 255)     
            
            sys.stdout.write('\r' + '{:03d}/{:03d}'.format(img_idx, len(img_path_l)))
            sys.stdout.flush()

            if img_idx >= border_frame and img_idx < max_idx - border_frame:  # center frames
                avg_psnr_center += crt_psnr
                N_center += 1
            else:  # border frames
                avg_psnr_border += crt_psnr
                N_border += 1
        avg_psnr = (avg_psnr_center + avg_psnr_border) / (N_center + N_border)
        avg_psnr_center = avg_psnr_center / N_center
        avg_psnr_border = 0 if N_border == 0 else avg_psnr_border / N_border
        avg_psnr_l.append(avg_psnr)
        avg_psnr_center_l.append(avg_psnr_center)
        avg_psnr_border_l.append(avg_psnr_border)

        logger.info('Folder {} - frames {} -  Average PSNR: {:.6f} dB; '
                    'Center PSNR: {:.6f} dB; '
                    'Border PSNR: {:.6f} dB.'
                    .format(subfolder_name, (N_center + N_border), avg_psnr,
                            avg_psnr_center, avg_psnr_border))
        break

    logger.info('################ Tidy Outputs ################')
    for i in range(len(subfolder_name_l)):
        logger.info('Folder {} - Average PSNR: {:.6f} dB, Center PSNR: {:.6f} dB, Border PSNR: {:.6f} dB. '
                    .format(subfolder_name_l[i], avg_psnr_l[i], avg_psnr_center_l[i], avg_psnr_border_l[i]))
    logger.info('################ Final Results ################')
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Total clips {}. Average PSNR: {:.6f} dB, Center PSNR: {:.6f} dB, Border PSNR: {:.6f} dB.'
                'Score: {:.6f}'.format(
                    len(subfolder_l),
                    sum(avg_psnr_l) / len(avg_psnr_l),
                    sum(avg_psnr_center_l) / len(avg_psnr_center_l),
                    sum(avg_psnr_border_l) / len(avg_psnr_border_l),
                    sum(avg_psnr_l) / len(avg_psnr_l) / 50))


if __name__ == '__main__':
    main()
