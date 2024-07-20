import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
import cv2
import scipy.io as sio
import logging
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

from hpf import *
from ResUNet1DB_conf3_learnp import UNet

from covnet import CovNet

TS_PARA = 60
PAYLOAD = 0.4

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def myParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='bows|bossbase|szu',default='szu')
    parser.add_argument('--dataroot', help='path to cover with the first sub-image being embedded',default='/data/ymx/SZU-cover-resample-256-C1stc-hill') 
    parser.add_argument('--coverroot', help='path to original cover dataset',default='/data/ymx/SZU-cover-resample-256') 
    parser.add_argument('--pmaproot', help='path to original probability map',default='/data/ymx/HILL_szu2/0.4') 
    
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')

    parser.add_argument('--niter', type=int, default=72, help='number of epochs to train for')
    parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate, default=0.0002')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)") 
    parser.add_argument('--netCov1', default='', help="path to netCov1 (to continue training)")
    parser.add_argument('--netCov2', default='', help="path to netCov2 (to continue training)")
    parser.add_argument('--outf', default='asym', help='folder to output images and model checkpoints')
    
    parser.add_argument('--config', default='G2_conf4_1102_ResUNet1DBconf3_learnp_covnet1e-6_C2adjust_D1D2_conf2lambda5_gl310_s3-1e-5_HILL', help='config for training') 
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    opt = parser.parse_args()
    return opt


class SZUDataset256(data.Dataset):
    def __init__(self, root, coverroot, pmaproot, transforms = None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

        self.img_path = root+'/{}.pgm'
        self.cover_path = coverroot+'/{}.pgm'
        self.pmat_path = pmaproot+'/{}_p.mat'

        self.transforms = transforms

    def __getitem__(self, index):
        # img_path = self.imgs[index]
        img_path = self.img_path.format(index+1)
        label = np.array([0, 1], dtype='int32')
        
        data = cv2.imread(img_path,-1)
        rand_ud = np.random.rand(256,256)

        cover_path = self.cover_path.format(index+1)
        ori_cover = cv2.imread(cover_path,-1)

        pmat_path = self.pmat_path.format(index+1)
        pmat = sio.loadmat(pmat_path)
        p_last = pmat['p']

        sample = {'data':data, 'rand_ud':rand_ud, 'label':label, 'ori_cover':ori_cover, 'p_last':p_last}
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

    def __len__(self):
        # return len(self.imgs)
        return 40000


class ToTensor():
  def __call__(self, sample):
    data, rand_ud, label, ori_cover, p_last = \
        sample['data'], sample['rand_ud'], sample['label'], sample['ori_cover'], sample['p_last']

    data = np.expand_dims(data, axis=0)
    data = data.astype(np.float32)
    rand_ud = rand_ud.astype(np.float32)
    rand_ud = np.expand_dims(rand_ud,axis = 0)

    ori_cover = np.expand_dims(ori_cover, axis=0)
    ori_cover = ori_cover.astype(np.float32)

    p_last = np.array(p_last)
    p_last = np.expand_dims(p_last, axis=0)
    p_last = p_last.astype(np.float32)

    # data = data / 255.0

    new_sample = {
      'data': torch.from_numpy(data),
      'rand_ud': torch.from_numpy(rand_ud),
      'label': torch.from_numpy(label).long(),
      'ori_cover': torch.from_numpy(ori_cover),
      'p_last': torch.from_numpy(p_last),
    }

    return new_sample



class Dataset256(data.Dataset):
    def __init__(self, image_dir,  transforms = None):

        self.image_dir = image_dir

        # self.prob_dir = prob_dir
        # self.dct_dir = dct_dir
        self.transforms = transforms
        self.postfix = 'resample-256-jpeg-75'
        self.steganography = 'bet-hill-uint8'

    def __getitem__(self ,index):
        #img_path = '{}-cover-resample-256/{}.pgm'.format(self.image_dir, str(index + 1))
        dct_mat_path = '{}-cover-dct-{}/{}.mat'.format(self.image_dir, self.postfix, str(index + 1))
        hill_mat_path = '{}-hill-cp-uint8-{}/{}.mat'.format(self.image_dir, self.postfix, str(index + 1))
        # bet_hill_mat_path = '{}-{}-0.4-pro-map-{}/{}.mat'.format(self.image_dir, self.steganography, self.postfix, str(index + 1))


        # label = np.array([0, 1], dtype='int32')
        #
        # data = cv2.imread(img_path ,-1)#0-255
        dct_mat = sio.loadmat(dct_mat_path)
        dct = dct_mat['C_COEFFS']  # -1024-1023
        nzac = dct_mat['nzAC']

        hill_mat = sio.loadmat(hill_mat_path)
        hill_cp = hill_mat['hill_cp']

        # prob_mat = sio.loadmat(bet_hill_mat_path)
        # prob = prob_mat['prob_map']#0-0.6



        rand_ud = np.random.rand(256 ,256)
        label = np.array([0, 1], dtype='int32')

        sample = {'dct':dct, 'nzac':nzac, 'hill_cp':hill_cp, 'rand_ud':rand_ud, 'label':label }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return 40000

class ToTensor256():
    def __call__(self, sample):
        dct, nzac, hill_cp, rand_ud, label = sample['dct'], sample['nzac'], sample['hill_cp'], sample['rand_ud'], sample['label']



        dct = np.expand_dims(dct, axis=0)
        dct = dct.astype(np.float32)

        nzac = nzac.astype(np.float32)

        hill_cp = np.expand_dims(hill_cp, axis=0)
        hill_cp = hill_cp.astype(np.float32)

        rand_ud = rand_ud.astype(np.float32)
        rand_ud = np.expand_dims(rand_ud,axis = 0)

        # prob = np.expand_dims(prob, axis=0)
        # prob = prob.astype(np.float32)



        # data = data / 255.0

        new_sample = {
            'dct': torch.from_numpy(dct),
            'nzac': torch.from_numpy(nzac),
            'hill_cp': torch.from_numpy(hill_cp),
            # 'prob': torch.from_numpy(prob),
            'rand_ud': torch.from_numpy(rand_ud),
            'label': torch.from_numpy(label).long(),

        }

        return new_sample


# custom weights initialization called on netG and netD
def weights_init_g(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def weights_init_d(net):
    for m in net.modules():
        if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
            m.weight.data.normal_(0., 0.01)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0., 0.01)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0., 0.02)
            m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

def weights_init_covnet(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.normal_(module.weight.data, mean=0, std=0.01)

      # nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')
      # nn.init.xavier_uniform_(module.weight.data)
      # nn.init.constant_(module.bias.data, val=0.2)
    # else:
    #   module.weight.requires_grad = True

  if type(module) == nn.Linear:
    nn.init.xavier_uniform_(module.weight.data)
    # nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)


def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss_Cov1']))

    y1 = hist['D_loss_Cov1']
    y2 = hist['D_loss_Cov2']
    y3 = hist['G_loss']

    
    plt.plot(x, y1, label='D_loss_Cov1')
    plt.plot(x, y2, label='D_loss_Cov2')
    plt.plot(x, y3, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    plt.savefig(path)

    plt.close()



def main():
    
    opt = myParseArgs()
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass


    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    LOG_PATH = os.path.join(opt.outf, 'model_log_'+opt.config)
    setLogger(LOG_PATH, mode = 'w')

    
    transform = transforms.Compose([ToTensor(),])
    dataset = SZUDataset256(opt.dataroot, opt.coverroot, opt.pmaproot, transforms=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers),drop_last = True)


    ############### Add texture reward ###############
    hpf = HPF_LAP2()
    hpf = nn.DataParallel(hpf)
    hpf = hpf.cuda()
    ############### Add texture reward ###############

    netG = UNet()
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    netG.apply(weights_init_g)


    if opt.netG != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netG))
        logging.info('-' * 8)
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    
    netCov1 = CovNet()
    netCov1 = nn.DataParallel(netCov1)
    netCov1 = netCov1.cuda()
    netCov1.apply(weights_init_covnet)

    if opt.netCov1 != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netCov1))
        logging.info('-' * 8)
        netCov1.load_state_dict(torch.load(opt.netCov1))
    # print(netCov1)

    netCov2 = CovNet()
    netCov2 = nn.DataParallel(netCov2)
    netCov2 = netCov2.cuda()
    netCov2.apply(weights_init_covnet)

    if opt.netCov2 != '':
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(opt.netCov2))
        logging.info('-' * 8)
        netCov2.load_state_dict(torch.load(opt.netCov2))
    # print(netCov2)

    criterion = nn.CrossEntropyLoss().cuda()

    # setup optimizer
    optimizerD1 = optim.Adam(netCov1.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerD2 = optim.Adam(netCov2.parameters(), lr=opt.lrD*0.1, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))

    scheduler_D1 = StepLR(optimizerD1, step_size=20, gamma=0.4)
    scheduler_D2 = StepLR(optimizerD2, step_size=20, gamma=0.4)
    scheduler_G = StepLR(optimizerG, step_size=20, gamma=0.4)
    
    train_hist = {}
    train_hist['D_loss_Cov1'] = []
    train_hist['D_loss_Cov2'] = []
    train_hist['G_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []

    ############### randomly divid ###############
    rnd = np.random.RandomState(1)

    pixelNum = opt.imageSize*opt.imageSize
    indices = np.arange(pixelNum)
    rnd.shuffle(indices)

    idxC1 = indices[:int(pixelNum/2)]
    idxC2 = indices[int(pixelNum/2):]
    # print(len(idxC1))

    maskC1 = np.zeros((pixelNum))
    maskC1[idxC1] = 1
    maskC2 = np.zeros((pixelNum))
    maskC2[idxC2] = 1

    maskC1 = maskC1.reshape(opt.imageSize,opt.imageSize)
    maskC1 = maskC1.astype(np.float32)
    maskC1 = torch.from_numpy(maskC1).cuda()
    maskC2 = maskC2.reshape(opt.imageSize,opt.imageSize)
    maskC2 = maskC2.astype(np.float32)
    maskC2 = torch.from_numpy(maskC2).cuda()
    # print(maskC1)
    ############### randomly divid ###############


    start_time = time.time()
    for epoch in range(opt.niter):

        netG.train()
        netCov1.train()
        netCov2.train()


        scheduler_D1.step()
        scheduler_D2.step()
        scheduler_G.step()


        epoch_start_time = time.time()
        for i,sample in enumerate(dataloader,0):
            
            # train with real
            optimizerG.zero_grad()

            data, rand_ud, label, ori_cover, p_last = \
                sample['data'], sample['rand_ud'],sample['label'],sample['ori_cover'],sample['p_last']
            cover, n, label, ori_cover, p_last = data.cuda(), rand_ud.cuda(), label.cuda(), ori_cover.cuda(), p_last.cuda()

            batch_maskC2 = maskC2.reshape(1,1,opt.imageSize,opt.imageSize).repeat(opt.batchSize,1,1,1).cuda()
            # print(batch_maskC2.shape)

            m1 = cover - ori_cover
            p_p, p_m = netG(m1, p_last)

            p_plus = p_p / 2.0 + 1e-5
            p_minus = p_m / 2.0 + 1e-5

            m = torch.zeros_like(cover)
            m[n < p_plus] = 1
            m[n > (torch.ones_like(cover)-p_minus)] = -1
            
            stego = cover + m * maskC2

            stego.requires_grad = True
            stego.retain_grad()

            C = -(p_plus * torch.log2(p_plus) + p_minus*torch.log2(p_minus)+ (1 - p_plus - p_minus +1e-5) * torch.log2(1 - p_plus - p_minus +1e-5))

            p_plus2 = p_plus * maskC2 + 1e-5
            p_minus2 = p_minus * maskC2 + 1e-5
            C2 = -(p_plus2 * torch.log2(p_plus2) + p_minus2*torch.log2(p_minus2)+ (1 - p_plus2 - p_minus2 +1e-5) * torch.log2(1 - p_plus2 - p_minus2 +1e-5))

        
            d_input1 = torch.zeros(opt.batchSize*2,1,256,256).cuda()
            d_input1[0:opt.batchSize*2:2,:] = cover
            d_input1[1:opt.batchSize*2:2,:] = stego

            d_input2 = torch.zeros(opt.batchSize*2,1,256,256).cuda()
            d_input2[0:opt.batchSize*2:2,:] = ori_cover
            d_input2[1:opt.batchSize*2:2,:] = stego

            final_m = stego - ori_cover
            changeRate = torch.mean(torch.abs(final_m))

            m2changeRate = torch.mean(torch.abs(m * maskC2))
            
            ############### Add texture reward ###############
            texture = hpf(cover)
            texture = abs(texture)
            texture = torch.sum(texture, dim = 0)
            # print(texture)
            ############### Add texture reward ###############
            
            label = label.reshape(-1)

            optimizerD1.zero_grad()
            errD1 = criterion(netCov1(d_input1), label) #.detach()
            
            errD1.backward(retain_graph=True)
            m_grad1 = stego.grad.data.clone() #用于更新G
            # print(m_grad1)

            errD1.backward()                
            optimizerD1.step()

            ################### add a new D #############
            stego.grad.data.zero_()

            optimizerD2.zero_grad()
            errD2 = criterion(netCov2(d_input2), label) #.detach()
            
            errD2.backward(retain_graph=True)
            m_grad2 = stego.grad.data.clone() #用于更新G

            errD2.backward()                
            optimizerD2.step()
            ################### add a new D #############
            lamda = 5
            m_grad = m_grad1 + lamda * m_grad2
            reward = 1e+7 * m * m_grad

            mask_plus = torch.where(m.eq(torch.ones_like(cover)),  
                                torch.ones_like(cover), torch.zeros_like(cover))
            mask_minus = torch.where(m.eq(-1 * torch.ones_like(cover)),
                                torch.ones_like(cover), torch.zeros_like(cover))
            entropy = torch.log(p_plus) * mask_plus + torch.log(p_minus) * mask_minus 
            g_l1 = torch.mean(entropy * reward * texture) 


            g_l2 = torch.mean((C2.sum(dim = (1,2,3)) - 256 * 256 * PAYLOAD / 2.0) ** 2)

            ################### reconstruction loss ###################
            g_l3 = torch.mean(torch.abs((p_last / 2.0 - p_plus) * maskC1) + torch.abs((p_last / 2.0 - p_minus) * maskC1))
            # print(g_l3.item())
            ################### reconstruction loss ###################

            errG = -g_l1 + 1e-6*g_l2 + 10*g_l3
            if epoch > 0:
                train_hist['G_loss'].append(errG.item())
                train_hist['D_loss_Cov1'].append(errD1.item())
                train_hist['D_loss_Cov2'].append(errD2.item())
            
            errG.backward()
            optimizerG.step()

            # iteration = epoch*1666+i
            # if iteration in [0, 1, 2, 100, 1000, 10000, 119000] : 
            #     torch.save(netG.state_dict(), '%s/netG_epoch_%s_%d_%d.pth' % (opt.outf, opt.config, epoch+1, iteration))

            logging.info('Epoch: [%d/%d][%d/%d] Loss_D1: %.4f Loss_D2: %.4f Loss_G: %.4f  C:%.4f  ChangeRate:%.4f m2ChangeRate:%.4f m1ChangeRate:%.4f' % 
            (epoch, opt.niter-1, i, len(dataloader), errD1.item(), errD2.item(), errG.item() ,C.sum().item()/opt.batchSize, changeRate.item(), m2changeRate.item(), changeRate.item()-m2changeRate.item()))
            
        train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            
        # do checkpointing
        if (epoch % 10 == 1 or epoch == opt.niter-1) and epoch >= 60 : 
            torch.save(netG.state_dict(), '%s/netG_epoch_%s_%d.pth' % (opt.outf, opt.config, epoch+1))

        loss_plot(train_hist, opt.outf, model_name = opt.outf + opt.config)


    train_hist['total_time'].append(time.time() - start_time)
    logging.info("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(train_hist['per_epoch_time']),
                                                                            epoch, train_hist['total_time'][0]))
    logging.info("Training finish!... save training results")

if __name__ == '__main__':
    main()