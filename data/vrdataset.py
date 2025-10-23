import pickle
from torch.utils.data import Dataset
import json
import numpy as np
import torchaudio
from .build import DATASET_REGISTRY
import torch.nn.functional as F
import torchvision.transforms.functional as F1
import os
import glob
import cv2
import pandas as pd
import time
import scipy.io as sio
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]

@DATASET_REGISTRY.register()
class Vrdataset(Dataset):

    def __init__(self, cfg, mode, dataset_flag=2):
        # VIEW_PATH = 'F:/dataset/vr_dataset/ep1_head_base_frame_16x9/'
        # VIEW_PATH = '/media/kemove/1A226EEF226ECEF7/work/pytorch_workplace/PARIMA-master/Viewport/'
        # basepath = 'F:/dataset/combine/videos'
        # Get the necessary information regarding the dimensions of the video
        # print("Dataset Reading JSON...")
        # file = open('E:/work/pytorch_workplace/avvp-main/mvit/datasets/meta.json', )
        # jsonRead = json.load(file)
        trj_dir_dataset1 = 'F:/dataset/vr_dataset/Gaze_txt_files'
        trj_dir_dataset2 = 'F:/dataset/gaze360/Gaze_txt_files'
        trj_dir_dataset3 = 'F:/dataset/ICME20/Gaze_txt_files'
        self.dataset1_vod = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9']
        #  icme 全部
        # self.time_start = {'1-1':7, '1-3':80, '1-4':15,  '1-7':25,  '1-9':35,'002':0, '003':30, '013':0, '014':0, '015':0, '019':0, '025':0, '026':0,'027':0, '028':0,'029':0
        #                    ,'033':0,'034':0,'040':10,'042':15,'044':25,'049':0,'050':2,'054':0,'058':0, '065':0, '067':0,  '070':0, '072':0, '083':10, '130':0, '133':15, '149':5, '151':0, '153':5, '168':10,
        #                    '5h95uTtPeck': 0, 'idLVnagjl_s': 0, '6QUCaLvQ_3I': 0, '8ESEI0bqrJ4': 0, '8feS1rNYEbg': 0, 'Bvu9m__ZX60': 0, 'ByBF08H-wDA': 0, 'dd39herpgXA': 0, 'ey9J7w98wlI': 0, 'fryDy9YcbI4': 0, 'gSueCRQO_5g': 0,
        #                    'idLVnagjl_s': 0, 'kZB3KMhqqyI': 0, 'MzcdEI-tSUc': 0, 'RbgxpagCY_c_2': 0}
        # self.time_end = {'1-1':45, '1-3':130,'1-4':45,  '1-7':45,  '1-9':80,'002':10, '003':40,'013':20,'014':20,'015':31,'019':30,'025':20,'026':20,'027':40,'028':30,'029':20
        #                  ,'033':35,'034':0,'040':50,'042':23,'044':45,'049':20,'050':10,'054':0,'058':25,'065':21,'067':40,'070':20, '072':40,'083':30,'130':36,'133':38, '149':25,'151':30, '153':25, '168':21,
        #                  '5h95uTtPeck':25, 'idLVnagjl_s':25, '6QUCaLvQ_3I':25, '8ESEI0bqrJ4':25, '8feS1rNYEbg':25, 'Bvu9m__ZX60':25, 'ByBF08H-wDA':25, 'dd39herpgXA':25, 'ey9J7w98wlI':25, 'fryDy9YcbI4':25, 'gSueCRQO_5g':25,
        #                    'idLVnagjl_s':25, 'kZB3KMhqqyI':25, 'MzcdEI-tSUc':25, 'RbgxpagCY_c_2':25}
        # 取消icme中 fps高的
        # self.time_start = {'1-1':7, '1-3':80, '1-4':15,  '1-7':25,  '1-9':35,'002':0, '003':30, '013':0, '014':0, '015':0, '019':0, '025':0, '026':0,'027':0, '028':0,'029':0
        #                    ,'033':0,'034':0,'040':10,'042':15,'044':25,'049':0,'050':2,'054':0,'058':0, '065':0, '067':0,  '070':0, '072':0, '083':10, '130':0, '133':15, '149':5, '151':0, '153':5, '168':10,
        #                    '5h95uTtPeck': 0, 'idLVnagjl_s': 0, '8ESEI0bqrJ4': 0, '8feS1rNYEbg': 0, 'Bvu9m__ZX60': 0, 'ByBF08H-wDA': 0, 'dd39herpgXA': 0, 'fryDy9YcbI4': 0, 'gSueCRQO_5g': 0,
        #                    'idLVnagjl_s': 0, 'kZB3KMhqqyI': 0, 'MzcdEI-tSUc': 0, 'RbgxpagCY_c_2': 0}
        # self.time_end = {'1-1':45, '1-3':130,'1-4':45,  '1-7':45,  '1-9':80,'002':10, '003':40,'013':20,'014':20,'015':31,'019':30,'025':20,'026':20,'027':40,'028':30,'029':20
        #                  ,'033':35,'034':0,'040':50,'042':23,'044':45,'049':20,'050':10,'054':0,'058':25,'065':21,'067':40,'070':20, '072':40,'083':30,'130':36,'133':38, '149':25,'151':30, '153':25, '168':21,
        #                  '5h95uTtPeck':25, 'idLVnagjl_s':25, '8ESEI0bqrJ4':25, '8feS1rNYEbg':25, 'Bvu9m__ZX60':25, 'ByBF08H-wDA':19, 'dd39herpgXA':25, 'fryDy9YcbI4':25, 'gSueCRQO_5g':25,
        #                    'idLVnagjl_s':25, 'kZB3KMhqqyI':25, 'MzcdEI-tSUc':25, 'RbgxpagCY_c_2':23}
        self.time_start = {'1-1': 25, '1-2': 15, '1-3': 110, '1-4': 25, '1-5': 35, '1-6': 132, '1-7': 5, '1-8': 66,
                                 '1-9': 15, '002': 1, '003': 30, '006': 1, '013': 1, '014': 1, '015': 1, '019': 0, '025': 1,
                                 '026': 0, '027': 0, '028': 1, '029': 1
            , '033': 0, '034': 0, '040': 10, '041': 0, '042': 15, '044': 25, '049': 0, '050': 2, '054': 0, '058': 0, '065': 0,
                                 '067': 0, '070': 1, '072': 1, '083': 10, '130': 0, '133': 18, '149': 5, '151': 0,
                                 '153': 5, '168': 10,
                                 '5h95uTtPeck': 0, 'idLVnagjl_s': 0, '8ESEI0bqrJ4': 0, '8feS1rNYEbg': 0,
                                 'Bvu9m__ZX60': 0, 'ByBF08H-wDA': 0, 'fryDy9YcbI4': 0, 'gSueCRQO_5g': 0,
                                 'kZB3KMhqqyI': 0, 'MzcdEI-tSUc': 0, 'RbgxpagCY_c_2': 0}
        self.time_end = {'1-1': 45, '1-2': 35, '1-3': 130, '1-4': 45, '1-5': 75, '1-6': 152, '1-7': 25, '1-8': 86,
                               '1-9': 35, '002': 9, '003': 40, '006': 9, '013': 18, '014': 17, '015': 28, '019': 30, '025': 18,
                               '026': 20, '027': 38, '028': 28, '029': 17
            , '033': 35, '034': 0, '040': 50, '041': 9, '042': 23, '044': 45, '049': 20, '050': 10, '054': 0, '058': 25,
                               '065': 21, '067': 40, '070': 17, '072': 18, '083': 30, '130': 36, '133': 37, '149': 25,
                               '151': 30, '153': 25, '168': 21,
                               '5h95uTtPeck': 25, 'idLVnagjl_s': 25, '8ESEI0bqrJ4': 25, '8feS1rNYEbg': 25,
                               'Bvu9m__ZX60': 25, 'ByBF08H-wDA': 19, 'dd39herpgXA': 25, 'fryDy9YcbI4': 25,
                               'gSueCRQO_5g': 25,
                               'kZB3KMhqqyI': 25, 'MzcdEI-tSUc': 25, 'RbgxpagCY_c_2': 23}
        #  eval = {'1-7','1-9', '033','054','058','065','067', '102', '130', '151', '168', 'idLVnagjl_s', 'kZB3KMhqqyI', '8ESEI0bqrJ4', '8feS1rNYEbg', 'ByBF08H-wDA', 'fryDy9YcbI4'}
        self.audio_dict = {}
        self.frame_start = {}
        self.frame_end = {}
        self.H = 144
        self.W = 256
        # self.frame_count = [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600]
        # self.width = jsonRead["dataset"][dataset_flag - 1]["width"]
        # self.height = jsonRead["dataset"][dataset_flag - 1]["height"]
        # self.view_width = jsonRead["dataset"][dataset_flag - 1]["view_width"]
        # self.view_height = jsonRead["dataset"][dataset_flag - 1]["view_height"]
        # self.milisec = jsonRead["dataset"][dataset_flag - 1]["milisec"]
        # self.gblur_size = 9
        self.look_back = cfg.hist_len
        # self.look_ahead = look_ahead
        self.n_col = cfg.GRID_W
        self.n_row = cfg.GRID_H
        self.videos_fps = np.load('E:/work/pytorch_workplace/avvp-main/mvit/datasets/video_fps.npy',  allow_pickle=True).item()
        self.process_frame_nums = cfg.process_frame_nums
        self.file_list = []
        for idx in self.time_start.keys():
            # start = int(self.frame_count[idx]/self.video_time[idx] * self.time_start[idx])
            # end = int(self.frame_count[idx] / self.video_time[idx] * self.time_end[idx])
            self.frame_start[idx] = int(self.time_start[idx] * self.videos_fps[idx])
            self.frame_end[idx] = int(self.time_end[idx] * self.videos_fps[idx])
        # get_Sample_List(trj_dir_dataset1)
        # get_Sample_List(trj_dir_dataset2)
        # vid_list = glob.glob(basepath+'/*')
        # user_file_count = 0

        self.get_Sample_List(trj_dir_dataset1, 1)
        self.get_Sample_List(trj_dir_dataset2, 2)
        # self.get_Sample_List(trj_dir_dataset3)

        # for vid_path in vid_list:
        #     odv_name = vid_path.split('\\')[1][:3]
        #     print(odv_name)
        #     if odv_name in self.dataset1_vod:
        #         trj_dir = 'F:/dataset/vr_dataset/Gaze_txt_files'
        #     else:
        #         trj_dir = 'F:/dataset/gaze360/Gaze_txt_files'

        #     user_list = glob.glob(trj_dir+'/*')
        #     for user_dir in user_list:
        #         print('user_dir=',user_dir)
        #         user_trj_list = glob.glob(user_dir + '/*')
        #         for user_trj_path in user_trj_list:
        #             print(user_trj_path)
        #             user_file_count+=1
        #             print('user file count=', user_file_count)
        #             aud_file_path = ''
        #             sal_file_path = trj_dir.replace('Gaze_txt_files', 'saliency')
        #             sal_file_path = os.path.join(sal_file_path, odv_name)
        #             label_file_path = ''
        #             # trj_logs = pd.read_table(user_trj_path, header=None, delimiter=',')
        #             start = self.frame_start.get(odv_name, 10)
        #             end = self.frame_end.get(odv_name, 500)
        #             user_trj_list = []
        #             for series_index in range(start, end - self.process_frame_nums, 15):
        #                 # print(series_index)
        #                 self.file_list.append(
        #                 [user_trj_path, sal_file_path, odv_name, series_index, aud_file_path])


    def __len__(self):
        """Denotes the total number of samples"""
        # print('total file=',len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        T0 = time.perf_counter()
        # print('index=', index)
        headmaps = []
        labelmaps = []
        headtrj = []
        labeltrj = []
        # print('end load view_info')
        sal_info = []
        audiowav = []
        sal_label = []
        bimaps = []
        sal_cube=[]
        batch = {}
        user_scan_path = self.file_list[index][0]
        sal_file_path = self.file_list[index][1]
        odv_name = self.file_list[index][2]
        fps = self.videos_fps[odv_name]
        series_index = self.file_list[index][3]
        aud_file_path = self.file_list[index][4]
        sal_label_path = self.file_list[index][5]
        bmap_path = self.file_list[index][6]
        user_name = self.file_list[index][7]
        start = series_index
        end = series_index + self.process_frame_nums

        trj_logs = pd.read_table(user_scan_path, header=None, delimiter=',')
        # print('user_sacan_path=', user_scan_path)
        # print('trj_logs.length=', len(trj_logs))
        Tstart = time.perf_counter()
        # sal_path = sal_file_path + '/{:07d}.pkl'.format(series_index)
        # sal_info = pickle.load(open(sal_path, "rb"), encoding="latin1")
        # sal_info = sal_info/255
        # T0 = time.perf_counter()
        # print('======sal time =', T0 - Tstart)
        frames_cube = []
        frames_equi = []
        frames = np.zeros((32, 40))
        for frame_index in range(start, end):
            # if ((frame_index - start) % 6 == 0) or (frame_index == end - 1):
            # if frame_index < start + self.look_back:
                # print((frame_index - start))
            # T1 = time.perf_counter()
            # if (end-frame_index) <= 16:    #  只取最后16帧图像进行salmap检测
            # if (frame_index - start) % 10 == 0:
            if frame_index >= 0:
                imgpath = sal_file_path + '/{:07d}.jpg'.format(frame_index)
                img = cv2.imread(imgpath)
                img = cv2.resize(img, (256, 144))
                frames_equi.append(img)

            # 加载用户轨迹
            try:
                geo_x = trj_logs.iloc[frame_index][3]
                geo_y = trj_logs.iloc[frame_index][4]
            except:
                print('frame_index=', frame_index, 'odv=', odv_name, ' user=', user_scan_path, 'trj_logs=',
                      trj_logs)
            if frame_index < start + self.look_back:
                trj_mat = np.zeros((self.H, self.W))
                head_x = int(geo_x * self.W)
                head_y = int(geo_y * self.H)
                if head_x == self.W:
                    head_x = self.W-1
                # print('x=', geo_x, 'y=', geo_y)
                # print('head_x=', head_x, 'head_y=', head_y)
                trj_mat[head_y, head_x] += 1
                headmaps.append(trj_mat)
                headtrj.append([geo_x,geo_y])
            else:
                label_mat = np.zeros((self.n_row, self.n_col))
                # print('x=',geo_x, 'y=',geo_y)
                label_x = int(geo_x * self.n_col)
                label_y = int(geo_y * self.n_row)
                if label_x == 16:
                    label_x = 15
                # print('label_x=', label_x, 'label_y=', label_y)
                label_mat[label_y, label_x] += 1
                labelmaps.append(label_mat)
                labeltrj.append([geo_x, geo_y])
        # print('odv=', odv_name,len(frames_equi))
        sal_info = torch.from_numpy(np.array(frames_equi)).permute(3, 0, 1, 2).float()
        # headtrj = np.array(headtrj).astype('float32')
        # labeltrj = np.array(labeltrj).astype('float32')
        batch['video'] = (sal_info / 255.)
        batch['vp_hist'] = np.array(headtrj).astype('float32')
        batch['target'] = np.array(labeltrj).astype('float32')
        batch['video_id'] = odv_name
        batch['user_id'] = user_name


        # labelmaps = labelmaps.reshape(labelmaps.shape[0], -1)

        return batch


    #  根据user轨迹
    def get_Sample_List(self, trj_dir, dataset):
        # evallist = ['102', '1-9','1-8','1-7','1-6']
        evallist = ['1-6', '1-7','1-8','1-9', '033','054','058','065','067', '102', '130', '151', '168', 'idLVnagjl_s', 'kZB3KMhqqyI', '8ESEI0bqrJ4', '8feS1rNYEbg', 'ByBF08H-wDA', 'fryDy9YcbI4']
        user_list = glob.glob(trj_dir + '/*')
        for user_dir in user_list:
            user_trj_list = glob.glob(user_dir + '/*')
            user_file_count = 0
            for user_trj_path in user_trj_list:
                # print(user_trj_path)
                if dataset == 1:
                    odv_name = user_trj_path.split('\\')[-1][:-4]
                elif dataset == 2:
                    odv_name = user_trj_path.split('\\')[-1][:3]
                user_name = user_trj_path.split('\\')[-2]
                if odv_name in evallist or odv_name not in self.time_start:
                    # print('odv_name = ',odv_name, '       skiped')
                    continue
                user_file_count += 1
                # print('user file count=', user_file_count)
                aud_file_path = ''
                # sal_file_path = trj_dir.replace('Gaze_txt_files', 'sample_frame_slice_90_256x320')
                sal_file_path = trj_dir.replace('Gaze_txt_files', 'frames')
                sal_file_path = os.path.join(sal_file_path, odv_name)
                sal_label_path = trj_dir.replace('Gaze_txt_files', 'saliency')
                sal_label_path = os.path.join(sal_label_path, odv_name)
                bmap_path = trj_dir.replace('Gaze_txt_files', 'fixation')
                bmap_path = os.path.join(bmap_path, odv_name)
                aud_file_path = trj_dir.replace('Gaze_txt_files', 'frames')
                aud_file_path = os.path.join(aud_file_path, odv_name) + '/' + odv_name +'.wav'
                label_file_path = ''
                trj_logs = pd.read_table(user_trj_path, header=None, delimiter=',')

                start = self.frame_start.get(odv_name, self.time_start.get(odv_name))
                end = self.frame_end.get(odv_name, self.time_end.get(odv_name))
                # start = 800
                # end = 1000
                if len(trj_logs)<end:
                    # end = len(trj_logs)-1
                    continue
                # print('odv_name = ',odv_name)
                for series_index in range(start, end - self.process_frame_nums - 10, 30):
                    # print(series_index)
                    self.file_list.append(
                        [user_trj_path, sal_file_path, odv_name, series_index,aud_file_path,sal_label_path,bmap_path, user_name])











