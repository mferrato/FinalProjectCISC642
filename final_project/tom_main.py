#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
from data_manager import Dataset, DataLoader, save_images
from models import UnetGenerator, VGGLoss, load_checkpoint

from visualization import save_images


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TOM")
    parser.add_argument('-b', type=int, default=4)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "train")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument("--max_step", type=int, default = 20000)

    opt = parser.parse_args()
    return opt

def train_tom(opt, loader, model, board):
    model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss()
    criterionMask = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda step: 1.0 -
            max(0, step - opt.max_step) / float((opt.max_step//2) + 1))

    for step in range(opt.max_step):
        iter_start_time = time.time()
        inputs = loader.next_batch()

        im = inputs['image'].cuda()
        agnostic = inputs['representation'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite+ p_rendered * (1 - m_composite)

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                    % (step+1, t, loss.item(), loss_l1.item(),
                    loss_vgg.item(), loss_mask.item()))

        if (step+1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step+1)))

def test_tom(opt, loader, model):
    model.cuda()
    model.eval()

    base_name = os.path.basename(opt.name)
    save_dir = os.path.join(opt.result_dir, base_name, opt.datamode)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try_on_dir = os.path.join(save_dir, 'try-on')
    if not os.path.exists(try_on_dir):
        os.makedirs(try_on_dir)
    print('Dataset size: %05d!' % (len(loader.dataset)))
    for step, inputs in enumerate(loader.data_loader):
        iter_start_time = time.time()

        person_names = inputs['person_name']
        agnostic = inputs['representation'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()

        outputs = model(torch.cat([agnostic, c],1))
        p_rendered, m_composite = torch.split(outputs, 3,1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        save_images(p_tryon, person_names, try_on_dir)

        if (step+1) % 50 == 0:
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f' % (step + 1, t))


def main():
    opt = get_opt()
    print(opt)
    print("TOM: Start to %s, named: %s!" % (opt.datamode, opt.name))

    # Dataset setup
    dataset = Dataset(opt)
    data_loader = DataLoader(opt, dataset, "TOM")

    model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)

    if opt.datamode == 'train':
        if not opt.checkpoint =='' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint)
        train_tom(opt, data_loader, model)
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_trained.pth'))
    elif opt.datamode == 'test':
        load_checkpoint(model, opt.checkpoint)
        with torch.no_grad():
            test_tom(opt, data_loader, model)
    else:
        raise NotImplementedError('Please input train or test stage')

    print('Finished test %s, named: %s!' % (opt.stage, opt.name))

if __name__ == "__main__":
    main()
