#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from data_manager import Dataset, DataLoader, save_images
from models import GMM, load_checkpoint, save_checkpoint


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument('-b', type=int, default=4)
    parser.add_argument("--stage", default = "train")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument("--max_step", type=int, default = 20000)

    opt = parser.parse_args()
    return opt

def train_gmm(opt, loader, model):
    model.cuda()
    model.train()

    criterionL1 = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.max_step) / float((opt.max_step//2) + 1))

    for step in range(opt.max_step):
        iter_start_time = time.time()
        inputs = loader.next_batch()

        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        im_c =  inputs['parse_cloth'].cuda()

        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')

        loss = criterionL1(warped_cloth, im_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Display loss update every 50 steps
        if (step + 1) % 50 == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step+1, t, loss.item()))

        # Save the model parameters every 500 steps, in case model crashes it can be resumed
        if (step + 1) % 500 == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)))

def test_gmm(opt, loader, model):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.name)
    save_dir = os.path.join("results", base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    warp_cloth_dir = os.path.join(save_dir, 'warp-cloth')
    if not os.path.exists(warp_cloth_dir):
        os.makedirs(warp_cloth_dir)
    warp_mask_dir = os.path.join(save_dir, 'warp-mask')
    if not os.path.exists(warp_mask_dir):
        os.makedirs(warp_mask_dir)

    for step, inputs in enumerate(loader.data_loader):
        iter_start_time = time.time()

        clothes_names = inputs['cloth_name']
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        grid, theta = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border')
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros')

        save_images(warped_cloth, clothes_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, clothes_names, warp_mask_dir)

        # Display inference update every 50 steps
        if (step + 1) % 50 == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step+1, t))

def main():
    opt = get_opt()
    print(opt)
    print("GMM: Start to %s, named: %s!" % (opt.stage, "GMM"))

    # dataset setup
    dataset = Dataset(opt, "GMM")
    dataset_loader = DataLoader(opt, dataset)

    model = GMM(opt)

    if opt.stage == 'train':
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_gmm(opt, dataset_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_trained.pth'))
    elif opt.stage == 'test':
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_gmm(opt, dataset_loader, model)
    else:
        raise NotImplementedError('Please input train or test stage')

    print('Finished %s stage, named: %s!' % (opt.datamode, opt.name))

if __name__ == "__main__":
    main()
