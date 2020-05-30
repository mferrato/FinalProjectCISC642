#coding=utf-8
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageDraw

import os.path as osp
import numpy as np
import json

class Dataset(data.Dataset):
    def __init__(self, opt, mod):
        super(Dataset, self).__init__()
        # base setting
        self.opt = opt
        self.root = "data"
        self.datamode = opt.datamode # train or test
        self.model = mod
        self.data_list = 'train_pairs.txt'
        if(self.datamode == 'test'):
            self.data_list = 'test_pairs.txt'
        self.width = 192
        self.height = 256
        self.r = 5 # radius
        self.data_path = osp.join(self.root, opt.datamode)
        self.transform = transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        # load data list
        person_names = []
        clothes_names = []
        with open(osp.join(self.root, self.data_list), 'r') as f:
            for line in f.readlines():
                person_name, cloth_name = line.strip().split()
                person_names.append(person_name)
                clothes_names.append(cloth_name)

        self.person_names = person_names
        self.clothes_names = clothes_names

    def name(self):
        return "Dataset"

    def __getitem__(self, index):
        cloth_name = self.clothes_names[index]
        person_name = self.person_names[index]


        if self.model == 'GMM':
            cloth = Image.open(osp.join(self.data_path, 'cloth', cloth_name))
            cloth_mask = Image.open(osp.join(self.data_path, 'cloth-mask', cloth_name))
        # Uses the generated cloth and cloth mask images (warped) by the GMM
        else:
            cloth = Image.open(osp.join(self.data_path, 'warp-cloth', cloth_name))
            cloth_mask = Image.open(osp.join(self.data_path, 'warp-mask', cloth_name))

        cloth = self.transform(cloth)
        cloth_mask_array = np.array(cloth_mask)
        cloth_mask_array = (cloth_mask_array >= 128).astype(np.float32)
        cloth_mask = torch.from_numpy(cloth_mask_array)
        cloth_mask.unsqueeze_(0)

        # person image
        person = Image.open(osp.join(self.data_path, 'image', person_name))
        person = self.transform(person)

        # load parsing image
        parse_name = person_name.replace('.jpg', '.png')
        person_parse = Image.open(osp.join(self.data_path, 'image-parse', parse_name))
        parse_array = np.array(person_parse)
        parse_shape = (parse_array > 0).astype(np.float32)
        parse_head = (parse_array == 1).astype(np.float32) + (parse_array == 2).astype(np.float32) + (parse_array == 4).astype(np.float32) + (parse_array == 13).astype(np.float32)
        parse_cloth = (parse_array == 5).astype(np.float32) + (parse_array == 6).astype(np.float32) + (parse_array == 7).astype(np.float32)


        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.width//16, self.height//16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.width, self.height), Image.BILINEAR)
        shape = self.transform(parse_shape)
        person_head = torch.from_numpy(parse_head)
        person_cloth_mask = torch.from_numpy(parse_cloth)


        im_c = person * person_cloth_mask + (1 - person_cloth_mask)
        im_h = person * person_head - (1 - person_head)

        # load pose points
        pose_name = person_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.height, self.width)
        im_pose = Image.new('L', (self.width, self.height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            temp_map = Image.new('L', (self.width, self.height))
            draw = ImageDraw.Draw(temp_map)
            x = pose_data[i,0]
            y = pose_data[i,1]
            if x > 1 and y > 1:
                draw.rectangle((x - self.r, y - self.r, x + self.r, y + self.r), 'white', 'white')
                pose_draw.rectangle((x - self.r, y - self.r, x + self.r, y + self.r), 'white', 'white')
            temp_map = self.transform(temp_map)
            pose_map[i] = temp_map[0]

        representation = torch.cat([shape, im_h, pose_map], 0)

        result = {
            'cloth_name':   cloth_name,
            'person_name':  person_name,
            'person': person,
            'cloth':    cloth,
            'cloth_mask':     cloth_mask,
            'agnostic': representation,
            'parse_cloth': im_c,    
            }

        return result

    def __len__(self):
        return len(self.person_names)

class DataLoader(object):
    def __init__(self, opt, dataset):
        super(DataLoader, self).__init__()

        train_sampler = torch.utils.data.sampler.RandomSampler(dataset)

        self.data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=4, shuffle=(train_sampler is None),
                num_workers=1, pin_memory=True, sampler=train_sampler)
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        Image.fromarray(array).save(osp.join(save_dir, img_name))
