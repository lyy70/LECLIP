import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    if dataset_name == 'mvtec_test':
        obj_list = ['wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'rsad':
        obj_list = ['mild_disease', 'moderate_disease', 'severe_disease', 'mix_disease']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'dagm':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'sdd':
        obj_list = ['sdd']
    elif dataset_name == 'dtd':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    # elif dataset_name == 'brainmri_cvpr':
    #     obj_list = ['BrainMRI']
    # elif dataset_name == 'liverct':
    #     obj_list = ['LiverCT']
    # elif dataset_name == 'resc':
    #     obj_list = ['RESC']
    # image-medical
    elif dataset_name == 'headct':
        obj_list = ['Head_CT']
    elif dataset_name == 'brainmri':
        obj_list = ['BrainMRI']
    elif dataset_name == 'br35H':
        obj_list = ['Br35H']
    elif dataset_name == 'covid-19':
        obj_list = ['COVID-19']
    elif dataset_name == 'ts':
        obj_list = ['TS']
    # pixel-medical
    elif dataset_name == 'isic':
        obj_list = ['ISIC']
    elif dataset_name == 'cvc-clinicdb':
        obj_list = ['CVC-ClinicDB']
    elif dataset_name == 'cvc-colondb':
        obj_list = ['CVC-ColonDB']
    elif dataset_name == 'kvasir':
        obj_list = ['Kvasir']
    elif dataset_name == 'endo':
        obj_list = ['Endo']
    elif dataset_name == 'tn3k':
        obj_list = ['TN3K']

    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        # pixel
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']

        # # # image
        # img_path, cls_name, specie_name, anomaly = data['img_path'], data['cls_name'], \
        #                                                       data['specie_name'], data['anomaly']

        img = Image.open(os.path.join(self.root, img_path))

        # pixel
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

        # transforms
        img = self.transform(img) if self.transform is not None else img
        # pixel
        img_mask = self.target_transform(
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}

        # ## image
        # return {'img': img, 'cls_name': cls_name, 'anomaly': anomaly,
        #         'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}

