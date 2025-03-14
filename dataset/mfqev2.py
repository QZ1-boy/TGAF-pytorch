import glob
import random
import torch
import os.path as op
import numpy as np
import os
# from cv2 import cv2
import cv2
from torch.utils import data as data
from utils import FileClient, paired_random_crop,  augment, totensor, import_yuv
import torch.nn.functional as F

def _bytes2img(img_bytes):
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = np.expand_dims(cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE), 2)  # (H W 1)
    img = img.astype(np.float32) / 255.
    return img


class MFQEv2Dataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        # print("{paired_random_crop img_gt}",img_gt.shape)
        # print("{paired_random_crop img_lqs}",img_lqs[0].shape)
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )
        
        # #  for scale x2
        # img_gt, img_lqs = paired_random_cropx2(
        #     img_gt, img_lqs, gt_size, img_gt_path
        #     )

        # print("{paired_random_crop img_gt}",img_gt.shape)
        # print("{paired_random_crop img_lqs}",img_lqs[0].shape)
        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)



class MFQEv2SRDataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        # print("{paired_random_crop img_gt}",img_gt.shape)
        # print("{paired_random_crop img_lqs}",img_lqs[0].shape)
        
        # randomly crop
        # img_gt, img_lqs = paired_random_crop(
        #     img_gt, img_lqs, gt_size, img_gt_path
        #     )
        
        # #  for scale x2
        img_gt, img_lqs = paired_random_cropx2(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # print("{paired_random_crop img_gt}",img_gt.shape)
        # print("{paired_random_crop img_lqs}",img_lqs[0].shape)
        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)




class MFQEv2PredDataset(data.Dataset):
    """MFQEv2 dataset.

    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        self.pred_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['pred_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        with open(self.meta_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.pred_root,
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'pred', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            self.neighbor_list = [i + (9 - nfs) // 2 for i in range(nfs)]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in self.neighbor_list:
            img_lq_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            img_lqs.append(img_lq)
        
        # get the neighboring pred frames
        img_preds = []
        for neighbor in self.neighbor_list:
            img_pred_path = f'{clip}/{seq}/im{neighbor}.png'
            img_bytes = self.file_client.get(img_pred_path, 'lq')
            img_pred = _bytes2img(img_bytes)  # (H W 1)
            img_preds.append(img_pred)

        # ==========
        # data augmentation
        # ==========
        # print("{paired_random_crop img_gt}",img_gt.shape)
        # print("{paired_random_crop img_lqs}",img_lqs[0].shape)
        
        # randomly crop
        # img_gt, img_lqs = paired_random_crop(
        #     img_gt, img_lqs, gt_size, img_gt_path
        #     )
        
        #  for scale x2
        img_gt, img_lqs, img_preds = paired_random_crop_predx2(
            img_gt, img_lqs, img_preds, gt_size, img_gt_path
            )

        # print("{paired_random_crop img_gt}",img_gt.shape)
        # print("{paired_random_crop img_lqs}",img_lqs[0].shape)
        # print("{paired_random_crop img_preds}",img_preds[0].shape)
        # flip, rotate
        img_lqs = img_lqs + img_preds  # gt joint augmentation with pred
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        # print("{paired_random_crop img_lqs}",type(img_lqs))
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        length = (len(img_results) -1) // 2
        # print("length",length)
        img_lqs = torch.stack(img_results[0:length], dim=0)
        img_preds = torch.stack(img_results[length:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'pred': img_preds,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)




class VideoTestMFQEv2Dataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        # assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict
        self.scale = 2  # opts_dict['scale']
        # print("{self.opts_dict['gt_path']}",self.opts_dict['gt_path'])

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        # print("gt_path_list",gt_path_list)
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_name_vid = name_vid
            # lq_name_vid = name_vid.replace(str(w),str(w//2))
            # lq_name_vid = lq_name_vid.replace(str(h),str(h//2))
            lq_vid_path = op.join(
                self.lq_root,
                lq_name_vid  #  lq_name_vid
                )
            # print("gt_vid_path",gt_vid_path)
            # print("lq_vid_path",lq_vid_path)
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # print("{img_gt}",img_gt.shape)
        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index] ,  #    // self.scale
                w=self.data_info['w'][index] ,  #    // self.scale
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            # print('[self.data_info[gt_path][index]]', self.data_info['gt_path'][index],self.data_info['gt_index'][index],self.data_info['lq_path'][index],lq_index)

            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor   #  需要修改 
        
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        
        # print("{img_lqs}",img_lqs[0].shape)

        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num




class VideoTestMFQEv2SRDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        # assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict
        self.scale = 2  # opts_dict['scale']
        # print("{self.opts_dict['gt_path']}",self.opts_dict['gt_path'])

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        # print("gt_path_list",gt_path_list)
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_name_vid = name_vid
            lq_name_vid = name_vid.replace(str(w),str(w//2))
            lq_name_vid = lq_name_vid.replace(str(h),str(h//2))
            lq_vid_path = op.join(
                self.lq_root,
                lq_name_vid  #  lq_name_vid
                )
            # print("gt_vid_path",gt_vid_path)
            # print("lq_vid_path",lq_vid_path)
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # print("{img_gt}",img_gt.shape)
        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index]// self.scale ,  #  
                w=self.data_info['w'][index]// self.scale ,  #  
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor   #  需要修改 
        
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]
        
        # print("{img_lqs}",img_lqs[0].shape)

        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num




class VideoTestMFQEv2PredDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        # assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict
        self.scale = 2  # opts_dict['scale']
        # print("{self.opts_dict['gt_path']}",self.opts_dict['gt_path'])

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        self.pred_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['pred_path']
            )
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'pred_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'pred_indexes': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        # print("gt_path_list",gt_path_list)
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            nfs_pred = nfs-1
            lq_name_vid = name_vid.replace(str(w),str(w//2))
            lq_name_vid = lq_name_vid.replace(str(h),str(h//2))
            pred_name_vid = lq_name_vid.replace(str(nfs),str(nfs_pred))   #####  问题  读取的不对
            # print("pred_name_vid",pred_name_vid)
            lq_vid_path = op.join(
                self.lq_root,
                lq_name_vid  #  lq_name_vid
                )
            pred_vid_path = op.join(
                self.pred_root,
                pred_name_vid  #  pred_name_vid
                )
            # print("gt_vid_path",len(gt_vid_path))
            # print("lq_vid_path",len(lq_vid_path))
            # print("pred_vid_path",len(pred_vid_path))
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                pred_indexes = list(range(iter_frm - radius - 1, iter_frm + radius))
                pred_indexes = list(np.clip(pred_indexes, 0, nfs - 2))
                # pred_indexes = list(np.clip(lq_indexes, 0, nfs - 2))
                # print("[lq_indexes]",lq_indexes)
                # print("[pred_indexes]",pred_indexes)
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['pred_path'].append(pred_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)
                self.data_info['pred_indexes'].append(pred_indexes)

    def __getitem__(self, index):
        # get gt frame
        # print('[index]',index)
        # print('[self.data_info[gt_path][index]]',self.data_info['gt_index'][index])

        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # print("{img_gt}",img_gt.shape)
        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            # print('[self.data_info[lq_path][index]]',lq_index)
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index] // self.scale,
                w=self.data_info['w'][index] // self.scale,
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            
            # if (self.data_info['h'][index] // self.scale % 8 != 0 ) or (self.data_info['w'][index] // self.scale % 8 != 0 ):
            #     # print("[ before self.lq]",self.lq.shape)
            #     img = img.squeeze(0)
            #     print("selflqqqq img",img.shape)
            #     print("selflqqqq img type",type(img))
            #     padder = InputPadder(img.shape)   #  , mode='sintel'
            #     img = padder.pad(img)
            #     img = img[0]  #  torch.Tensor(
            #     # print("[self.lq]",type(self.lq))
            #     img = img.unsqueeze(0)
            
            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)
        # print("img_lqs",len(img_lqs))
        # get pred frames    ### [[[have problem]]]
        img_preds = []
        for pred_index in self.data_info['pred_indexes'][index]:
            # print("{self.data_info['pred_path'][index]}",self.data_info['pred_path'][index])
            # print("[pred_index]",pred_index)
            # if 
            img = import_yuv(
                seq_path=self.data_info['pred_path'][index],
                h=self.data_info['h'][index] // self.scale,
                w=self.data_info['w'][index] // self.scale,
                tot_frm=1,
                yuv_type='400p',
                start_frm=pred_index,
                only_y=True
                )
            
            
            # print("[ before 000 img]",img.shape)
            # if (self.data_info['h'][index] // self.scale % 8 != 0 ) or (self.data_info['w'][index] // self.scale % 8 != 0 ):
            #     print("[ before img.lq]",img.shape)
            #     img = img.squeeze(0)
            #     # print("selflqqqq",type(self.lq))
            #     padder = InputPadder(img.shape)   #  , mode='sintel'
            #     img = padder.pad(img)
            #     img = img[0]  #  torch.Tensor(
            #     # print("[self.lq]",type(self.lq))
            #     img = img.unsqueeze(0)           
            
            img_pred = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_preds.append(img_pred)
        # img_preds.insert(0,img_lqs[0])
        # print("img_preds",len(img_preds))

        # no any augmentation

        # to tensor   #  需要修改 
        # print("img_preds",type(img_preds))
        # print("img_gt",type(img_gt))
        img_lqs = img_lqs +  img_preds
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        length = (len(img_results)-1) // 2
        img_lqs = torch.stack(img_results[0:length], dim=0)
        img_preds = torch.stack(img_results[length:-1], dim=0)
        img_gt = img_results[-1]
        
        # print("{img_lqs}",img_lqs[0].shape)

        return {
            'lq': img_lqs,  # (T 1 H W)
            'pred': img_preds,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num




class VideoTestMFQEv2PredframeDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.

    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        # assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict
        self.scale = 2  # opts_dict['scale']
        # print("{self.opts_dict['gt_path']}",self.opts_dict['gt_path'])

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        self.pred_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['pred_path']
            )
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'pred_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'pred_indexes': [],
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        # print("gt_path_list",gt_path_list)
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            nfs_pred = nfs-1
            lq_name_vid = name_vid.replace(str(w),str(w//2))
            lq_name_vid = lq_name_vid.replace(str(h),str(h//2))
            pred_name_vid = lq_name_vid.replace(str(nfs),str(nfs_pred))   #####  问题  读取的不对
            # print("pred_name_vid",pred_name_vid)
            lq_vid_path = op.join(
                self.lq_root,
                lq_name_vid  #  lq_name_vid
                )
            pred_vid_path = op.join(
                self.pred_root,
                pred_name_vid  #  pred_name_vid
                )
            # print("gt_vid_path",len(gt_vid_path))
            # print("lq_vid_path",len(lq_vid_path))
            # print("pred_vid_path",len(pred_vid_path))
            for iter_frm in range(nfs):
                lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                pred_indexes = list(range(iter_frm - radius - 1, iter_frm + radius))
                pred_indexes = list(np.clip(pred_indexes, 0, nfs - 2))
                # pred_indexes = list(np.clip(lq_indexes, 0, nfs - 2))
                # print("[lq_indexes]",lq_indexes)
                # print("[pred_indexes]",pred_indexes)
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['pred_path'].append(pred_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)
                self.data_info['pred_indexes'].append(pred_indexes)

    def __getitem__(self, index):
        # get gt frame
        # print('[index]',index)
        # print('[self.data_info[gt_path][index]]',self.data_info['gt_index'][index])

        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # print("{img_gt}",img_gt.shape)
        # get lq frames
        img_lqs = []
        # for lq_index in self.data_info['lq_indexes'][index]:
        lq_index = self.data_info['lq_indexes'][index]

            # print('[self.data_info[lq_path][index]]',lq_index)
        img = import_yuv(
            seq_path=self.data_info['lq_path'][index],
            h=self.data_info['h'][index] // self.scale,
            w=self.data_info['w'][index] // self.scale,
            tot_frm=1,
            start_frm=lq_index,
            only_y=True
            )
        
        # if (self.data_info['h'][index] // self.scale % 8 != 0 ) or (self.data_info['w'][index] // self.scale % 8 != 0 ):
        #     # print("[ before self.lq]",self.lq.shape)
        #     img = img.squeeze(0)
        #     print("selflqqqq img",img.shape)
        #     print("selflqqqq img type",type(img))
        #     padder = InputPadder(img.shape)   #  , mode='sintel'
        #     img = padder.pad(img)
        #     img = img[0]  #  torch.Tensor(
        #     # print("[self.lq]",type(self.lq))
        #     img = img.unsqueeze(0)
        
        img_lq = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)
        # img_lqs.append(img_lq)
        # print("img_lqs",len(img_lqs))
        # get pred frames    ### [[[have problem]]]
        # img_preds = []
        # for pred_index in self.data_info['pred_indexes'][index]:
        pred_index = self.data_info['pred_indexes'][index]

            # print("{self.data_info['pred_path'][index]}",self.data_info['pred_path'][index])
            # print("[pred_index]",pred_index)
            # if 
        img = import_yuv(
            seq_path=self.data_info['pred_path'][index],
            h=self.data_info['h'][index] // self.scale,
            w=self.data_info['w'][index] // self.scale,
            tot_frm=1,
            yuv_type='400p',
            start_frm=pred_index,
            only_y=True
            )
            
            
            # print("[ before 000 img]",img.shape)
            # if (self.data_info['h'][index] // self.scale % 8 != 0 ) or (self.data_info['w'][index] // self.scale % 8 != 0 ):
            #     print("[ before img.lq]",img.shape)
            #     img = img.squeeze(0)
            #     # print("selflqqqq",type(self.lq))
            #     padder = InputPadder(img.shape)   #  , mode='sintel'
            #     img = padder.pad(img)
            #     img = img[0]  #  torch.Tensor(
            #     # print("[self.lq]",type(self.lq))
            #     img = img.unsqueeze(0)           
            
        img_pred = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)
        # img_preds.append(img_pred)
        # img_preds.insert(0,img_lqs[0])
        # print("img_preds",len(img_preds))

        # no any augmentation

        # to tensor   #  需要修改 
        # print("img_preds",type(img_preds))
        # print("img_gt",type(img_gt))
        img_lq = img_lq +  img_pred
        img_lq.append(img_gt)
        img_results = totensor(img_lq)
        length = (len(img_results)-1) // 2
        img_lqs = torch.stack(img_results[0:length], dim=0)
        img_preds = torch.stack(img_results[length:-1], dim=0)
        img_gt = img_results[-1]
        
        # print("{img_lqs}",img_lqs[0].shape)
        # print("{img_preds}",img_preds[0].shape)

        return {
            'lq': img_lqs,  # (T 1 H W)
            'pred': img_preds,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num




class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
            # self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def unpadx2(self,x):
        ht, wd = x.shape[-2:]
        c = [2*self._pad[2], ht-2*self._pad[3], 2*self._pad[0], wd-2*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def unpadx3(self,x):
        ht, wd = x.shape[-2:]
        c = [3*self._pad[2], ht-3*self._pad[3], 3*self._pad[0], wd-3*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

    def unpadx4(self,x):
        ht, wd = x.shape[-2:]
        c = [4*self._pad[2], ht-4*self._pad[3], 4*self._pad[0], wd-4*self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]




class MFQEv2HQDataset(data.Dataset):
    """MFQEv2 dataset.
    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        black_list = []
        v_index2total_f = []
        for i in range(109):
            v_index2total_f.append(0)
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                v = int(v)
                f = int(f)
                v_index2total_f[v] += 1
        self.keys = []
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                if(v in black_list):continue
                f=int(f)
                v=int(v)
                if(f<radius+1):continue
                if(f>v_index2total_f[v]-radius):continue
                self.keys.append(tmp)
        self.v_index2total_f = v_index2total_f
        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        nfs = self.v_index2total_f[int(clip)]
        neighbor_list = f2list(int(seq), nfs)
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{str(neighbor).zfill(3)}/'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            # try:
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # except:
            #     print(seq,nfs,neighbor_list)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)



class VideoTestMFQEv2HQDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 dataset recommended by ITU-T.
    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
                )
            # print("NFS=",nfs)
            for iter_frm in range(nfs):
                # lq_indexes = list(range(iter_frm - radius, iter_frm + radius + 1))
                lq_indexes = f2list_valid(iter_frm,nfs)
                lq_indexes = list(np.clip(lq_indexes, 0, nfs - 1))
                # print("input",iter_frm)
                # print("get",lq_indexes)
                self.data_info['index_vid'].append(idx_vid)
                self.data_info['gt_path'].append(gt_vid_path)
                self.data_info['lq_path'].append(lq_vid_path)
                self.data_info['name_vid'].append(name_vid)
                self.data_info['w'].append(w)
                self.data_info['h'].append(h)
                self.data_info['gt_index'].append(iter_frm)
                self.data_info['lq_indexes'].append(lq_indexes)
            # os._exit(233)
        # print(len(self.data_info['gt_path']),"233")

    def __getitem__(self, index):
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=1,
            start_frm=self.data_info['gt_index'][index],
            only_y=True
            )
        img_gt = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1)

        # get lq frames
        img_lqs = []
        for lq_index in self.data_info['lq_indexes'][index]:
            img = import_yuv(
                seq_path=self.data_info['lq_path'][index],
                h=self.data_info['h'][index],
                w=self.data_info['w'][index],
                tot_frm=1,
                start_frm=lq_index,
                only_y=True
                )
            img_lq = np.expand_dims(
                np.squeeze(img), 2
                ).astype(np.float32) / 255.  # (H W 1)
            img_lqs.append(img_lq)

        # no any augmentation

        # to tensor
        img_lqs.append(img_gt)
        img_results = totensor(img_lqs)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gt,  # (1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return len(self.data_info['gt_path'])

    def get_vid_num(self):
        return self.vid_num



# 一个标准的全部做进去的LMDB
class MFQEv2BetaDataset(data.Dataset):
    """MFQEv2Beta dataset.
    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        self.radius = radius
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        black_list = []
        v_index2total_f = []
        for i in range(109):
            v_index2total_f.append(0)
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                v = int(v)
                f = int(f)
                v_index2total_f[v] += 1
        self.keys = []
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                if(v in black_list):continue
                f=int(f)
                v=int(v)
                if(f<radius+1):continue
                if(f>v_index2total_f[v]-radius):continue
                self.keys.append(tmp)
        self.v_index2total_f = v_index2total_f
        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gt_path = key
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = _bytes2img(img_bytes)  # (H W 1)

        # get the neighboring LQ frames
        img_lqs = []
        nfs = self.v_index2total_f[int(clip)]
        # neighbor_list = f2list(int(seq), nfs)
        neighbor_list = list(range(int(seq) - self.radius, int(seq) + self.radius + 1))
        neighbor_list = list(np.clip(neighbor_list, 0, nfs - 1))
        # print(seq,'to',neighbor_list)
        # os._exit(233)
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{str(neighbor).zfill(3)}/'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            # try:
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # except:
            #     print(seq,nfs,neighbor_list)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gt, img_lqs = paired_random_crop(
            img_gt, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs.append(img_gt)  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gt,  # ([RGB] H W)
            }

    def __len__(self):
        return len(self.keys)



class MFQEv2RTDataset(data.Dataset):
    """MFQEv2RT dataset.
    For training data: LMDB is adopted. See create_lmdb for details.
    
    Return: A dict includes:
        img_lqs: (T, [RGB], H, W)
        img_gt: ([3, RGB], H, W)
        key: str
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        self.opts_dict = opts_dict
        self.radius = radius
        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )

        # extract keys from meta_info.txt
        self.meta_info_path = op.join(
            self.gt_root, 
            self.opts_dict['meta_info_fp']
            )
        black_list = []
        v_index2total_f = []
        for i in range(109):
            v_index2total_f.append(0)
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                v = int(v)
                f = int(f)
                v_index2total_f[v] += 1
        self.keys = []
        with open(self.meta_info_path, 'r') as fin:
            for line in fin:
                tmp = line.split(' ')[0]
                v,f,pos=tmp.split("/")
                if(v in black_list):continue
                f=int(f)
                v=int(v)
                if(f<radius+1):continue
                if(f>v_index2total_f[v]-radius):continue
                self.keys.append(tmp)
        self.v_index2total_f = v_index2total_f
        # define file client
        self.file_client = None
        self.io_opts_dict = dict()  # FileClient needs
        self.io_opts_dict['type'] = 'lmdb'
        self.io_opts_dict['db_paths'] = [
            self.lq_root, 
            self.gt_root
            ]
        self.io_opts_dict['client_keys'] = ['lq', 'gt']

        # generate neighboring frame indexes
        # indices of input images
        # radius | nfs | input index
        # 0      | 1   | 4, 4, 4  # special case, for image enhancement
        # 1      | 3   | 3, 4, 5
        # 2      | 5   | 2, 3, 4, 5, 6 
        # 3      | 7   | 1, 2, 3, 4, 5, 6, 7
        # no more! septuplet sequences!
        if radius == 0:
            self.neighbor_list = [4, 4, 4]  # always the im4.png
        else:
            nfs = 2 * radius + 1
            if radius == 4:
                PAD = 1
            else:
                PAD = 0
            self.neighbor_list = [PAD+i + (9 - nfs) // 2 for i in range(nfs)]
            # print("@@",self.neighbor_list)
            # os._exit(233)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_opts_dict.pop('type'), **self.io_opts_dict
            )
        # random reverse
        if self.opts_dict['random_reverse'] and random.random() < 0.5:
            self.neighbor_list.reverse()

        # ==========
        # get frames
        # ==========

        # get the GT frame (im4.png)
        gt_size = self.opts_dict['gt_size']
        key = self.keys[index]
        clip, seq, _ = key.split('/')  # key example: 00001/0001/im1.png

        img_gts = []
        for i in range(-self.radius,self.radius+1):
            img_gt_path = str(clip)+'/'+str(int(seq)+i).zfill(3)+'/'
            try:
                img_bytes = self.file_client.get(img_gt_path, 'gt')
                img_gt = _bytes2img(img_bytes)  # (H W 1)
            except:
                print("Fail to get",img_gt_path)
                print("EZ to get",key)
                os._exit(233)
            img_gts.append(img_gt)
        # get the neighboring LQ frames
        img_lqs = []
        nfs = self.v_index2total_f[int(clip)]
        # neighbor_list = f2list(int(seq), nfs)
        neighbor_list = list(range(int(seq) - self.radius, int(seq) + self.radius + 1))
        neighbor_list = list(np.clip(neighbor_list, 0, nfs - 1))
        # print(seq,'to',neighbor_list)
        # os._exit(233)
        for neighbor in neighbor_list:
            img_lq_path = f'{clip}/{str(neighbor).zfill(3)}/'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            # print("??",img_lq_path)
            # try:
            img_lq = _bytes2img(img_bytes)  # (H W 1)
            # except:
            #     print(seq,nfs,neighbor_list)
            img_lqs.append(img_lq)

        # ==========
        # data augmentation
        # ==========
        
        # randomly crop
        img_gts, img_lqs = paired_random_crop(
            img_gts, img_lqs, gt_size, img_gt_path
            )

        # flip, rotate
        img_lqs = img_lqs + img_gts  # gt joint augmentation with lq
        img_results = augment(
            img_lqs, self.opts_dict['use_flip'], self.opts_dict['use_rot']
            )

        # to tensor
        img_results = totensor(img_results)
        L = len(img_results)//2
        img_lqs = torch.stack(img_results[0:-L], dim=0)
        img_gts = torch.stack(img_results[-L:], dim=0)

        return {
            'lq': img_lqs,  # (T [RGB] H W)
            'gt': img_gts,  # (3 [RGB] H W)
            }

    def __len__(self):
        return len(self.keys)



class VideoTestMFQEv2RTDataset(data.Dataset):
    """
    Video test dataset for MFQEv2 RT dataset recommended by ITU-T.
    For validation data: Disk IO is adopted.
    
    Test all frames. For the front and the last frames, they serve as their own
    neighboring frames.
    """
    def __init__(self, opts_dict, radius):
        super().__init__()

        assert radius != 0, "Not implemented!"
        
        self.opts_dict = opts_dict

        # dataset paths
        self.gt_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['gt_path']
            )
        self.lq_root = op.join(
            'data/MFQEv2/', 
            self.opts_dict['lq_path']
            )
        
        # record data info for loading
        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'gt_index': [], 
            'lq_indexes': [], 
            'h': [], 
            'w': [], 
            'index_vid': [], 
            'name_vid': [], 
            'nfs': [],
            }
        gt_path_list = sorted(glob.glob(op.join(self.gt_root, '*.yuv')))
        self.vid_num = len(gt_path_list)
        for idx_vid, gt_vid_path in enumerate(gt_path_list):
            name_vid = gt_vid_path.split('/')[-1]
            w, h = map(int, name_vid.split('_')[-2].split('x'))
            nfs = int(name_vid.split('.')[-2].split('_')[-1])
            lq_vid_path = op.join(
                self.lq_root,
                name_vid
                )
            self.data_info['w'].append(w)
            self.data_info['h'].append(h)
            self.data_info['gt_path'].append(gt_vid_path)
            self.data_info['lq_path'].append(lq_vid_path)
            self.data_info['nfs'].append(nfs)
            self.data_info['name_vid'].append(name_vid)
            self.data_info['index_vid'].append(idx_vid)

    def __getitem__(self, index):
        nfs = self.data_info['nfs'][index]
        # nfs = 100
        # get gt frame
        img = import_yuv(
            seq_path=self.data_info['gt_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=nfs,
            start_frm=0,
            only_y=True
            )
        img_gts = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1 T)
        # print(img_gt.shape)
        # os._exit(233)
        # get lq frames
        img = import_yuv(
            seq_path=self.data_info['lq_path'][index],
            h=self.data_info['h'][index],
            w=self.data_info['w'][index],
            tot_frm=nfs,
            start_frm=0,
            only_y=True
            )
        img_lqs = np.expand_dims(
            np.squeeze(img), 2
            ).astype(np.float32) / 255.  # (H W 1 T)

        # no any augmentation
        img_gts = img_gts.astype('float32')
        img_lqs = img_lqs.astype('float32')
        # to tensor
        # print(img_gts.shape,'vs_1',img_lqs.shape)
        # os._exit(233)
        img_gts = torch.from_numpy(img_gts.transpose((0,2,1,3)))
        img_lqs = torch.from_numpy(img_lqs.transpose((0,2,1,3)))
        # print(img_gts.size(),'vs_2',img_lqs.size())
        return {
            'lq': img_lqs,  # (T 1 H W)
            'gt': img_gts,  # (T 1 H W)
            'name_vid': self.data_info['name_vid'][index], 
            'index_vid': self.data_info['index_vid'][index], 
            }

    def __len__(self):
        return self.vid_num

    def get_vid_num(self):
        return self.vid_num