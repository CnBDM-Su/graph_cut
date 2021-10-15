from top_n_similarity import generate_similarity
from generate_edge import generate_graph_edge
from info import infomap
from evaluate import evaluation, single_evaluation
from combine_level2 import combine
import os.path as osp
import numpy as np
import pandas as pd
import time

class Graph_cut:
    def __init__(self,
                 mode='',
                 data_path='',
                 output_path='',
                 similarity_file_name='',
                 edge_file_name='',
                 label_file_name='',
                 face_threshold=0.6,
                 body_threshold=0.7,
                 face_name='',
                 body_name='',
                 fb_name='',
                 fb_edge_path=''):

        self.mode = mode
        self.data_path = data_path
        self.output_path = output_path
        self.face_threshold = face_threshold
        self.body_threshold = body_threshold
        self.similarity_path = osp.join(self.data_path, mode, similarity_file_name)
        self.edge_path = osp.join(self.output_path, mode, edge_file_name)
        self.label_path = osp.join(self.output_path, mode, label_file_name)
        self.capturetime_path = osp.join(self.data_path, mode, 'all_'+ mode + '_capturetime_v3.npy')
        self.deviceip_path = osp.join(self.data_path, mode, 'all_' + mode + '_deviceip_v3.npy')
        self.imgname_path = osp.join(self.data_path, mode, 'all_' + mode + '_globalimgname_v3.npy')
        self.full_path = osp.join(self.data_path, mode, 'all_' + mode + '_fullpath_v3.npy')
        self.gt_path = osp.join(self.data_path, mode, 'all_' + mode + '_label_v3.npy')
        self.feature_path = osp.join(self.data_path, mode, 'all_' + mode + '_feature_v3.npy')
        self.face_quality_path = osp.join(self.data_path, mode, 'all_' + mode + '_all_face_quality_v3.npy')

        self.face_path = osp.join(self.output_path, 'face', face_name)
        self.body_path = osp.join(self.output_path, 'body', body_name)
        self.fb_edge_path = osp.join(self.data_path, fb_edge_path)
        self.fb_pred_path = osp.join(self.output_path, 'face_body', fb_name)

        self.res_label = 1
        self.face_num = np.load(osp.join(self.data_path, 'face', 'all_face_label_v3.npy')).shape[0]
        self.body_num = np.load(osp.join(self.data_path, 'body', 'all_body_label_v3.npy')).shape[0]
        self.face_body_num = np.load(osp.join(self.data_path, 'face', 'all_face_label_v3.npy')).shape[0] + np.load(osp.join(self.data_path, 'body', 'all_body_label_v3.npy')).shape[0]

        if mode == 'face':
            self.threshold = self.face_threshold
            self.num = self.face_num
        elif mode == 'body':
            self.threshold = self.body_threshold
            self.num = self.body_num
        elif mode == 'face_body':
            self.num = self.face_body_num
        else:
            print('Wrong Mode')

    def graph_generate(self):
        path = self.similarity_path
        if not osp.exists(path):
            t = time.perf_counter()
            print('generating ' + self.mode + ' similarity')
            print('Done! {} s'.format(str(time.perf_counter() - t)[:4]))

        path = self.edge_path
        if not osp.exists(path):
            t = time.perf_counter()
            print('generating ' + self.mode + ' edge')
            generate_graph_edge(self.threshold, self.similarity_path, self.imgname_path, self.edge_path)
            print('Done! {} s'.format(str(time.perf_counter() - t)[:4]))

    def graph_cut(self):
        path = self.label_path
        if not osp.exists(path):
            t = time.perf_counter()
            print('generating ' + self.mode + ' pred labels')
            infomap(self.num, self.edge_path, self.label_path)
            print('Done! {} s'.format(str(time.perf_counter() - t)[:4]))

    def face_body_combine(self):
        if not osp.exists(self.fb_edge_path):
            self.face_body_combine()
        path = self.fb_pred_path
        if not osp.exists(path):
            combine(self.face_path, self.body_path, self.face_quality_path, self.fb_edge_path, self.fb_pred_path)
        self.res_label = 3

    def evaluate(self):
        path_map = {1:self.label_path, 3:self.fb_pred_path}
        evaluate_path = path_map[self.res_label]

        evaluation(self.gt_path, evaluate_path)

    def face_body_edge(self):
        face = np.load(osp.join(self.data_path, 'face', 'all_face_globalimgname_v3.npy'))
        num = face.shape[0]
        body = np.load(osp.join(self.data_path, 'body', 'all_body_globalimgname_v3.npy'))
        for i in range(face.shape[0]):
            face[i] = '-'.join(face[i].split('-')[:3])
        for i in range(body.shape[0]):
            body[i] = '-'.join(body[i].split('-')[:3])

        face = pd.DataFrame(face).reset_index()
        face.columns = ['id_f', 'track']
        body = pd.DataFrame(body).reset_index()
        body.columns = ['id_b', 'track']

        body.id_b = body.id_b.astype(int) + num

        fb_edge = []
        fb = pd.merge(face, body)
        for i in range(fb.shape[0]):
            fb_edge.append([fb.id_f[i], fb.id_b[i]])

        fb_edge = np.array(fb_edge).T
        ones = np.ones(shape=(1, fb_edge.shape[1]))
        fb_edge = np.concatenate([fb_edge, ones], 0)
        np.save(self.fb_edge_path, fb_edge)