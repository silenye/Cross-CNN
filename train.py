import torch
import torch.nn as nn
import torchvision
from TransUNet_model.TransUnet import VisionTransformer
import argparse
from TransUNet_model.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from vgg_loss import VGGPerceptualLoss
from my_dataset import My_Dataset_train
from my_dataset import My_Dataset_test
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np


# unnormalize
def unnormalized_show(img):
    img = img * 0.5 + 0.5
    npimg = img.detach().cpu().numpy()
    # plt.figure()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return torch.from_numpy(npimg)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='../data/Synapse/train_npz', help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=3, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=1000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()


    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = VisionTransformer(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()


    # 定义损失
    L1_loss = nn.L1Loss()

    # 定义数据集
    train_dataset = My_Dataset_train(args.img_size)
    test_dataset = My_Dataset_test(args.img_size)

    # 定义dataloader
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=3, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=True)

    # 定义优化器
    optimizer = torch.optim.Adam(net.parameters(), 1e-4, weight_decay=1e-6)

    train_step = 0
    test_step = 0

    for epoch_num in range(0, args.max_epochs):
        net.train()
        loop_train = tqdm(train_loader, total=len(train_loader))
        for sampled_batch in loop_train:
            ref, skt, gt = sampled_batch
            ref, skt, gt = ref.cuda(), skt.cuda(), gt.cuda()

            input = torch.cat((ref, skt), dim=1).cuda()
            output = net(input)


            l1_loss_train = L1_loss(output, gt)
            total_loss_train = 2 * l1_loss_train

            optimizer.zero_grad()
            total_loss_train.backward()
            optimizer.step()

            loop_train.set_description(f'Epoch_train [{epoch_num}/{args.max_epochs}]')
            loop_train.set_postfix(loss=total_loss_train.item())


        # test
        l1_loss_test_list = []
        vgg_loss_test_list = []
        total_loss_test_list = []
        psnr_list = []
        ssim_list = []

        net.eval()

        loop_test = tqdm(test_loader, total=len(test_loader))
        for sampled_batch in loop_test:
            test_step += 1
            ref, skt, gt = sampled_batch
            ref, skt, gt = ref.cuda(), skt.cuda(), gt.cuda()

            input = torch.cat((ref, skt), dim=1).cuda()
            with torch.no_grad():
                output = net(input)

            for j in range(gt.shape[0]):
                psnr = -10 * math.log10(torch.mean((gt[j] - output[j]) * (gt[j] - output[j])).cpu().data)
                psnr_list.append(psnr)
                # ssim = ssim_matlab(gt[j].unsqueeze(0), output[j].unsqueeze(0))
                # ssim_list.append(ssim.cpu().numpy())

            loop_test.set_description(f'Epoch_test [{epoch_num}/{args.max_epochs}]')

        model_save_path = ''
        torch.save(net.state_dict(), model_save_path)

