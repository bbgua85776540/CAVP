import pickle
from torch.utils.data import Dataset
import json
import numpy as np
import torchaudio
from .build import DATASET_REGISTRY
import torch.nn.functional as F
import os
import glob
import cv2
import pandas as pd
import scipy.io as sio
from PIL import Image
import torch

MEAN = [ 110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0 ]
STD = [ 38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0 ]

@DATASET_REGISTRY.register()
class Evaldataset(Dataset):

    def __init__(self, cfg, mode, dataset_flag=2):
        VIEW_PATH = 'F:/dataset/vr_dataset/ep1_head_base_frame_16x9/'
        # VIEW_PATH = '/media/kemove/1A226EEF226ECEF7/work/pytorch_workplace/PARIMA-master/Viewport/'
        basepath = 'F:/dataset/combine/videos'
        # Get the necessary information regarding the dimensions of the video
        # print("Dataset Reading JSON...")
        file = open('E:/work/pytorch_workplace/avvp-main/mvit/datasets/meta.json', )
        jsonRead = json.load(file)
        trj_dir_dataset1 = 'F:/dataset/vr_dataset/Gaze_txt_files'
        trj_dir_dataset2 = 'F:/dataset/gaze360/Gaze_txt_files'
        trj_dir_dataset3 = 'F:/dataset/ICME20/Gaze_txt_files'
        # print('dataset_flag=', dataset_flag)
        # self.fps = [29, 29, 30, 29, 29, 29, 25, 25, 29]  # need repaire
        # self.frame_count = [4921, 5994, 8797, 5172, 6165, 19632, 11251, 4076, 8603]
        # self.dataset1_vod = ['1-1', '1-2', '1-3', '1-4', '1-5', '1-6', '1-7', '1-8', '1-9']
        # self.frame_count = {'1-1':4921, '1-2':5994, '1-3':8797, '1-4':5172, '1-5':6165, '1-6':19632, '1-7':11251, '1-8':4076, '1-9':8603}
        # self.video_time = {'1-1':164, '1-2':201, '1-3':293, '1-4':172, '1-5':205, '1-6':655, '1-7':451, '1-8':164, '1-9':292}
        # self.time_start = {'1-1':7, '1-3':80, '1-4':15,  '1-7':25,  '1-9':35,'002':0, '003':30, '013':0, '014':0, '015':0, '019':0, '025':0, '026':0,'027':0, '028':0,'029':0
        #                    ,'033':0,'034':0,'040':10,'042':15,'044':25,'049':0,'050':2,'054':0,'058':0, '065':0, '067':0,  '070':0, '072':0, '083':10, '130':0, '133':15, '149':5, '151':0, '153':5, '168':10,
        #                    '5h95uTtPeck': 0, 'idLVnagjl_s': 0, '8ESEI0bqrJ4': 0, '8feS1rNYEbg': 0, 'Bvu9m__ZX60': 0, 'ByBF08H-wDA': 0, 'dd39herpgXA': 0, 'fryDy9YcbI4': 0, 'gSueCRQO_5g': 0,
        #                    'idLVnagjl_s': 0, 'kZB3KMhqqyI': 0, 'MzcdEI-tSUc': 0, 'RbgxpagCY_c_2': 0}
        # self.time_end = {'1-1':45, '1-3':130,'1-4':45,  '1-7':45,  '1-9':80,'002':10, '003':40,'013':20,'014':20,'015':31,'019':30,'025':20,'026':20,'027':40,'028':30,'029':20
        #                  ,'033':35,'034':0,'040':50,'042':23,'044':45,'049':20,'050':10,'054':0,'058':25,'065':21,'067':40,'070':20, '072':40,'083':30,'130':36,'133':38, '149':25,'151':30, '153':25, '168':21,
        #                  '5h95uTtPeck':25, 'idLVnagjl_s':25, '8ESEI0bqrJ4':25, '8feS1rNYEbg':25, 'Bvu9m__ZX60':25, 'ByBF08H-wDA':19, 'dd39herpgXA':25, 'fryDy9YcbI4':25, 'gSueCRQO_5g':25,
        #                    'idLVnagjl_s':25, 'kZB3KMhqqyI':25, 'MzcdEI-tSUc':25, 'RbgxpagCY_c_2':23}
        self.time_start = {'1-1': 7, '1-2': 5, '1-3': 80, '1-4': 15, '1-5': 45, '1-6': 152, '1-7': 25, '1-8': 86,
                                '1-9': 35, '002': 0, '003': 30, '013': 0, '014': 0, '015': 0, '019': 0, '025': 0,
                                '026': 0, '027': 0, '028': 0, '029': 0
            , '033': 0, '034': 0, '040': 10, '042': 15, '044': 25, '049': 0, '050': 2, '054': 0, '058': 0, '065': 0,
                                '067': 0, '070': 0, '072': 0, '083': 10, '130': 0, '133': 15, '149': 5, '151': 0,
                                '153': 5, '168': 10,
                                '5h95uTtPeck': 0, 'idLVnagjl_s': 0, '8ESEI0bqrJ4': 0, '8feS1rNYEbg': 0,
                                'Bvu9m__ZX60': 0, 'ByBF08H-wDA': 0, 'fryDy9YcbI4': 0, 'gSueCRQO_5g': 0,
                                'kZB3KMhqqyI': 0, 'MzcdEI-tSUc': 0, 'RbgxpagCY_c_2': 0}
        self.time_end = {'1-1': 45, '1-2': 35, '1-3': 130, '1-4': 45, '1-5': 75, '1-6': 182, '1-7': 45,
                              '1-8': 116, '1-9': 80, '002': 10, '003': 40, '013': 20, '014': 20, '015': 31, '019': 30,
                              '025': 20, '026': 20, '027': 40, '028': 30, '029': 20
            , '033': 35, '034': 0, '040': 50, '042': 23, '044': 45, '049': 20, '050': 10, '054': 0, '058': 25,
                              '065': 21, '067': 40, '070': 20, '072': 40, '083': 30, '130': 36, '133': 38, '149': 25,
                              '151': 30, '153': 25, '168': 21,
                              '5h95uTtPeck': 25, 'idLVnagjl_s': 25, '8ESEI0bqrJ4': 25, '8feS1rNYEbg': 25,
                              'Bvu9m__ZX60': 25, 'ByBF08H-wDA': 19, 'dd39herpgXA': 25, 'fryDy9YcbI4': 25,
                              'gSueCRQO_5g': 25,
                              'kZB3KMhqqyI': 25, 'MzcdEI-tSUc': 25, 'RbgxpagCY_c_2': 23}
        self.audio_dict = {}
        self.frame_start = {}
        self.frame_end = {}
        self.H = 144
        self.W = 256
        # self.frame_count = [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600]
        self.width = jsonRead["dataset"][dataset_flag - 1]["width"]
        self.height = jsonRead["dataset"][dataset_flag - 1]["height"]
        self.view_width = jsonRead["dataset"][dataset_flag - 1]["view_width"]
        self.view_height = jsonRead["dataset"][dataset_flag - 1]["view_height"]
        self.milisec = jsonRead["dataset"][dataset_flag - 1]["milisec"]
        # self.gblur_size = 9
        self.look_back = cfg.hist_len
        # self.look_ahead = look_ahead
        self.n_col = cfg.GRID_W
        self.n_row = cfg.GRID_H
        self.process_frame_nums = cfg.process_frame_nums
        self.videos_fps = np.load('E:/work/pytorch_workplace/avvp-main/mvit/datasets/video_fps.npy',
                                  allow_pickle=True).item()
        self.file_list = []
        for idx in self.time_start.keys():
            self.frame_start[idx] = int(self.time_start[idx] * self.videos_fps[idx])
            self.frame_end[idx] = int(self.time_end[idx] * self.videos_fps[idx])
        # get_Sample_List(trj_dir_dataset1)
        # get_Sample_List(trj_dir_dataset2)
        vid_list = glob.glob(basepath+'/*')
        user_file_count = 0

        self.get_Sample_List(trj_dir_dataset1, 1)
        self.get_Sample_List(trj_dir_dataset2, 2)
        # self.get_Sample_List(trj_dir_dataset3)




    def __len__(self):
        """Denotes the total number of samples"""
        # print('total file=',len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        # print('index=', index)
        # if index==138968:
        #     print('1111111')
        headmaps = []
        labelmaps = []
        # print('end load view_info')
        sal_info = []
        audiowav = []
        sal_label = []
        headtrj = []
        labeltrj = []
        bimaps = []
        sal_cube = []
        batch = {}
        user_scan_path = self.file_list[index][0]
        sal_file_path = self.file_list[index][1]
        series_index = self.file_list[index][3]
        odv_name = self.file_list[index][4]
        user_name = self.file_list[index][5]
        aud_file_path = self.file_list[index][6]
        odv_name = self.file_list[index][2]
        sal_label_path = self.file_list[index][7]
        bmap_path = self.file_list[index][8]
        fps = self.videos_fps[odv_name]
        start = series_index
        end = series_index + self.process_frame_nums

        trj_logs = pd.read_table(user_scan_path, header=None, delimiter=',')
        frames_equi = []
        frames = np.zeros((32, 40))
        for frame_index in range(start, end):
            # if ((frame_index - start) % 6 == 0) or (frame_index == end - 1):

            # if (end - frame_index) <= 16:
            # if (frame_index - start) % 10 == 0:
            # if frame_index < start + self.look_back:
            if frame_index >= 0:

                imgpath = sal_file_path + '/{:07d}.jpg'.format(frame_index)

                # with open(imgpath, 'rb') as f:
                #     with Image.open(f) as img:
                img = cv2.imread(imgpath)
                # print(imgpath)
                img = cv2.resize(img, (256, 144))
                frames_equi.append(img)

            # 加载用户轨迹
            try:
                geo_x = trj_logs.iloc[frame_index][3]
                geo_y = trj_logs.iloc[frame_index][4]
            except:
                print('user_scan_path=',user_scan_path,'    frame index =' ,frame_index)
            if frame_index < start + self.look_back:
                trj_mat = np.zeros((self.H, self.W))
                head_x = int(geo_x * self.W)
                head_y = int(geo_y * self.H)
                if head_x == self.W:
                    head_x = self.W-1
                trj_mat[head_y, head_x] += 1
                headmaps.append(trj_mat)
                headtrj.append([geo_x, geo_y])
            else:
                label_mat = np.zeros((self.n_row, self.n_col))
                # print('x=',geo_x, 'y=',geo_y)
                label_x = int(geo_x * self.n_col)
                label_y = int(geo_y * self.n_row)
                if label_x == 16:
                    label_x = 15
                label_mat[label_y, label_x] += 1
                labelmaps.append(label_mat)
                labeltrj.append([geo_x, geo_y])


        sal_info = torch.from_numpy(np.array(frames_equi)).permute(3, 0, 1, 2).float()
        batch['video'] = (sal_info / 255.)

        batch['vp_hist'] = np.array(headtrj).astype('float32')
        batch['target'] = np.array(labeltrj).astype('float32')
        batch['user_id'] = user_name
        batch['video_id'] = odv_name
        batch['series_index'] = series_index
        # headtrj = np.array(headtrj).astype('float32')
        # labeltrj = np.array(labeltrj).astype('float32')


        return batch

    def get_Sample_List(self, trj_dir, dataset):
        # evallist = ['1-8','1-6','215','214','213','212','211','210','209','208','206','205','204','203','202','201','200','199'
        #             ,'198','197','196','195','194','193','192','191']
        # evallist = ['1-8', '1-6', '54', '67', '215', '214', '213', '212', '211', '210', '209', '208', '206',
        #             '205', '204', '203', '202', '201', '200', '199'
        #     , '198', '197', '196', '195', '194', '193', '192', '191']
        evallist = ['1-6', '1-7','1-8','1-9', '033','054','058','065','067', '130', '151', '168', 'idLVnagjl_s', 'kZB3KMhqqyI', '8ESEI0bqrJ4', '8feS1rNYEbg', 'ByBF08H-wDA', 'fryDy9YcbI4']
        # evallist = ['ByBF08H-wDA', 'fryDy9YcbI4','1-6','1-7','1-8', '1-9']
        # evallist = ['ByBF08H-wDA', 'fryDy9YcbI4', '1-7']
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
                if odv_name not in evallist:
                    continue
                # print('eval odv_name = ', odv_name)
                user_file_count += 1
                # print('user file count=', user_file_count)
                aud_file_path = ''
                sal_file_path = ''
                # sal_file_path = trj_dir.replace('Gaze_txt_files', 'sample_frame_slice_90_256x320')
                # sal_file_path = os.path.join(sal_file_path, odv_name)
                sal_file_path = trj_dir.replace('Gaze_txt_files', 'frames')
                sal_file_path = os.path.join(sal_file_path, odv_name)
                sal_label_path = trj_dir.replace('Gaze_txt_files', 'saliency')
                sal_label_path = os.path.join(sal_label_path, odv_name)
                bmap_path = trj_dir.replace('Gaze_txt_files', 'fixation')
                bmap_path = os.path.join(bmap_path, odv_name)
                aud_file_path = trj_dir.replace('Gaze_txt_files', 'frames')
                aud_file_path = os.path.join(aud_file_path, odv_name) + '/' + odv_name +'.wav'
                label_file_path = ''
                # trj_logs = pd.read_table(user_trj_path, header=None, delimiter=',')
                start = self.frame_start.get(odv_name, self.time_start.get(odv_name))
                end = self.frame_end.get(odv_name, self.time_end.get(odv_name))
                for series_index in range(start, end - self.process_frame_nums, self.process_frame_nums - self.look_back):
                    # print(series_index)
                    self.file_list.append(
                        [user_trj_path, sal_file_path, odv_name, series_index, odv_name, user_name,aud_file_path,sal_label_path, bmap_path])




