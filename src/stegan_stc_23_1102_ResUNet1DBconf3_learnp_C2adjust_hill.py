import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import os


from PIL import Image
import time
import cv2
import scipy.io as sio
from pathlib import Path

from ResUNet1DB_conf3_learnp import UNet
import os

WETCOST = 10e+10

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,0,1'

def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--netG', default='./asym/netG_epoch_G2_conf4_1102_ResUNet1DBconf3_learnp_covnet1e-6_C2adjust_D1D2_conf2lambda5_gl310_s3-1e-5_HILL_0.4_72.pth', help="path to netG2 (to continue training)")

    parser.add_argument('--config', default='G2_conf4_1102_ResUNet1DBconf3_learnp_covnet1e-6_C2adjust_D1D2_conf2lambda5_gl310_s3-1e-5_HILL_retrain0.4_72', help="train result")  

    parser.add_argument('--datacover', help='path to cover with the first sub-image being embedded',default='/data/ymx/BB-cover-resample-256-C1stc-hill-0.4/') 
    parser.add_argument('--pmaproot', help='path to probability map dataset',default='/data/ymx/HILL_bb/0.4/') 
    parser.add_argument('--coverroot', help='path to original cover dataset',default='/data/ymx/BB-cover-resample-256-testset/') 
    parser.add_argument('--indexpath', help='path to index',default='./index_list/boss_gan_train_1w.npy')
    
    parser.add_argument('--root', default='/data/ymx/asym_GMAN_240107_globalTable/', help="path to save result")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    
    parser.add_argument('--TS_PARA', type=int, default=1000000, help='parameter for double tanh')
    parser.add_argument('--payload', type=float, default=0.4, help='embedding rate')  #0.4
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    
    args = parser.parse_args()
    return args


def ternary_entropyf(pP1, pM1):
    p0 = 1-pP1-pM1
    P = torch.hstack((p0.flatten(), pP1.flatten(), pM1.flatten()))
    H = -P*torch.log2(P)
    eps = 2.2204e-16
    H[P<eps] = 0
    H[P>1-eps] = 0
    return torch.sum(H)

def calc_lambda(rho_p1, rho_m1, message_length, n):
    l3 = 1e+3
    m3 = float(message_length+1)
    iterations = 0

    while m3 > message_length:
        l3 = l3 * 2
        pP1 = (torch.exp(-l3 * rho_p1)) / (1 + torch.exp(-l3 * rho_p1) + torch.exp(-l3 * rho_m1))
        pM1 = (torch.exp(-l3 * rho_m1)) / (1 + torch.exp(-l3 * rho_p1) + torch.exp(-l3 * rho_m1))
        m3 = ternary_entropyf(pP1, pM1)

        iterations += 1
        if iterations > 10:
            return l3
    l1 = 0
    m1 = float(n)
    lamb = 0

    # iterations = 0
    alpha = float(message_length)/n
    # limit search to 30 iterations and require that relative payload embedded
    # is roughly within 1/1000 of the required relative payload
    while float(m1-m3)/n > alpha/1000.0 and iterations<30:  #300
        lamb = l1+(l3-l1)/2
        pP1 = (torch.exp(-lamb*rho_p1))/(1+torch.exp(-lamb*rho_p1)+torch.exp(-lamb*rho_m1))
        pM1 = (torch.exp(-lamb*rho_m1))/(1+torch.exp(-lamb*rho_p1)+torch.exp(-lamb*rho_m1))
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lamb
            m3 = m2
        else:
            l1 = lamb
            m1 = m2
    iterations = iterations + 1
    return lamb

def embedding_simulator(batch_x, batch_rho_p1, batch_rho_m1, m, seed):
    batch_y = torch.zeros(batch_x.shape).cuda()
    for i in range(batch_x.shape[0]):
        x = batch_x[i,0,:]
        rho_p1 = batch_rho_p1[i,0,:]
        rho_m1 = batch_rho_m1[i,0,:]

        n = x.shape[0]
        lamb = calc_lambda(rho_p1, rho_m1, m, n)
        pChangeP1 = (torch.exp(-lamb * rho_p1)) / (1 + torch.exp(-lamb * rho_p1) + torch.exp(-lamb * rho_m1))
        pChangeM1 = (torch.exp(-lamb * rho_m1)) / (1 + torch.exp(-lamb * rho_p1) + torch.exp(-lamb * rho_m1))

        y = x.clone()
        #print("seed: "+str(seed[i]))
        torch.manual_seed(seed[i])
        randChange = torch.rand(y.shape[0]).cuda()

        y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1
        y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1
        batch_y[i,0,:] = y
    
    # batch_y = batch_y.astype(np.float32)
    # batch_y = torch.from_numpy(batch_y)
    return batch_y

def embedding_simulator_img(batch_x, batch_rho_p1, batch_rho_m1, m, seed):
    batch_y = torch.zeros(batch_x.shape).cuda()
    for i in range(batch_x.shape[0]):
        x = batch_x[i,0,:,:]
        rho_p1 = batch_rho_p1[i,0,:,:]
        rho_m1 = batch_rho_m1[i, 0, :, :]

        n = x.shape[0]*x.shape[1]
        lamb = calc_lambda(rho_p1, rho_m1, m, n)
        pChangeP1 = (torch.exp(-lamb * rho_p1)) / (1 + torch.exp(-lamb * rho_p1) + torch.exp(-lamb * rho_m1))
        pChangeM1 = (torch.exp(-lamb * rho_m1)) / (1 + torch.exp(-lamb * rho_p1) + torch.exp(-lamb * rho_m1))

        y = x.clone()
        #print("seed: "+str(seed[i]))
        torch.manual_seed(seed[i])
        randChange = torch.rand([y.shape[0], y.shape[1]]).cuda()

        y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1
        y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] = y[(randChange >= pChangeP1) & (randChange < pChangeP1+pChangeM1)] - 1
        batch_y[i,0,:,:] = y
    
    # batch_y = batch_y.astype(np.float32)
    # batch_y = torch.from_numpy(batch_y)
    return batch_y


class Dataset(data.Dataset):
    def __init__(self, cover_path, ori_cover_path, pmat_path, index_path, transforms = None):
        self.index_list = np.load(index_path)
        self.cover_path = cover_path+'{}.pgm'
        self.pmat_path = pmat_path+'{}_p.mat'

        self.ori_cover_path = ori_cover_path+'{}.pgm'

        self.transforms = transforms        
        

    def __getitem__(self,index):
        file_index = self.index_list[index]
        cover_path = self.cover_path.format(file_index)
        cover = cv2.imread(cover_path, -1)
        label = np.array([0, 1], dtype='int32')
        
        pmat_path = self.pmat_path.format(file_index)
        pmat = sio.loadmat(pmat_path)
        p_last = pmat['p']

        ori_cover_path = self.ori_cover_path.format(file_index)
        ori_cover = cv2.imread(ori_cover_path, -1)

        rand_ud = np.random.rand(256, 256)
        sample = {'cover':cover, 'rand_ud':rand_ud, 'label':label, 'index':file_index, 'p_last':p_last, 'ori_cover':ori_cover}
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

        
    def __len__(self):
        return len(self.index_list)


class ToTensor():
  def __call__(self, sample):
    cover, rand_ud, label, index, p_last, ori_cover  = \
        sample['cover'], sample['rand_ud'], sample['label'], sample['index'], sample['p_last'], sample['ori_cover']

    cover = np.expand_dims(cover, axis=0)
    cover = cover.astype(np.float32)
    
    rand_ud = rand_ud.astype(np.float32)
    rand_ud = np.expand_dims(rand_ud,axis = 0)
    
    p_last = np.array(p_last)
    # p_last = np.expand_dims(p_last, axis=0)
    p_last = p_last.astype(np.float32)

    ori_cover = np.expand_dims(ori_cover, axis=0)
    ori_cover = ori_cover.astype(np.float32)

    new_sample = {
      'cover': torch.from_numpy(cover),
      'rand_ud': torch.from_numpy(rand_ud),
      'label': torch.from_numpy(label).long(),
      'index':index,
      'p_last': torch.from_numpy(p_last),
      'ori_cover':torch.from_numpy(ori_cover)
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


def adjust_bn_stats(model, train_loader):
  model.train()

  with torch.no_grad():
    for i,sample in enumerate(train_loader,0):

        cover, rand_ud, label, index= sample['cover'], sample['rand_ud'], sample['label'], sample['index']
        cover, n, label = cover.cuda(), rand_ud.cuda(), label.cuda()

        # learn the probas
        p = model(cover)


def main():
    
    args = myParseArgs()
    try:
        stego_path = os.path.join(args.root, args.config)
        os.makedirs(stego_path)
    except OSError:
        pass

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True
        
    
    transform = transforms.Compose([ToTensor(),])
    dataset = Dataset(args.datacover, args.coverroot, args.pmaproot, args.indexpath, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers),drop_last = True)


    netG = UNet()
    netG = nn.DataParallel(netG)
    netG = netG.cuda()
    #netG.apply(weights_init_g)
    
    if args.netG != '':
        print('Load netG state_dict in {}'.format(args.netG))
        netG.load_state_dict(torch.load(args.netG))

    ###################### set mask C1 and C2 ###################
    rnd = np.random.RandomState(1)

    pixelNum = args.imageSize*args.imageSize
    indices = np.arange(pixelNum)
    rnd.shuffle(indices)

    idxC1 = indices[:int(pixelNum/2)]
    idxC2 = indices[int(pixelNum/2):]
    # print(idxC1)

    maskC1 = np.zeros((pixelNum))
    maskC1[idxC1] = 1
    maskC2 = np.zeros((pixelNum))
    maskC2[idxC2] = 1

    maskC1 = maskC1.reshape(args.imageSize,args.imageSize)
    maskC1 = maskC1.astype(np.float32)
    maskC1 = torch.from_numpy(maskC1).cuda()
    maskC2 = maskC2.reshape(args.imageSize,args.imageSize)
    maskC2 = maskC2.astype(np.float32)
    maskC2 = torch.from_numpy(maskC2).cuda()
    ###################### set mask C1 and C2 ###################
    changeRate = 0

    netG.eval()
    with torch.no_grad():
        for i,sample in enumerate(dataloader,0):

            cover, rand_ud, label, index, p_last, ori_cover = \
                sample['cover'], sample['rand_ud'], sample['label'], sample['index'],sample['p_last'], sample['ori_cover']
            cover, n, label, p_last, ori_cover = cover.cuda(), rand_ud.cuda(), label.cuda(), p_last.cuda(), ori_cover.cuda()

            # batch_maskC2 = maskC2.reshape(1,1,args.imageSize,args.imageSize).repeat(args.batchSize,1,1,1).cuda()

            m1 = cover - ori_cover
            p_p, p_m = netG(m1, p_last)

            p_plus = p_p / 2.0 + 1e-5
            p_minus = p_m / 2.0 + 1e-5
            
            rho_plus = torch.log(1/p_plus-2)
            rho_minus = torch.log(1/p_minus-2)
            
            ########################## extract C2 part ###########
            rho_plus_C2 = torch.flatten(rho_plus, start_dim=2, end_dim=3)[:,:,idxC2]
            rho_minus_C2 = torch.flatten(rho_minus, start_dim=2, end_dim=3)[:,:,idxC2]
            cover_C2 = torch.flatten(cover, start_dim=2, end_dim=3)[:,:,idxC2]
            # print(cover_C2.shape)
            ########################## extract C2 part ###########

            ######################## STC embedding #############
            message_length = args.imageSize*args.imageSize*args.payload / 2.0

            rho_plus_C2[rho_plus_C2 > WETCOST] = WETCOST
            rho_plus_C2[torch.isinf(rho_plus_C2)] = WETCOST
            rho_minus_C2[rho_minus_C2 > WETCOST] = WETCOST
            rho_minus_C2[torch.isinf(rho_minus_C2)] = 10e+10

            rho_plus_C2[cover_C2==255] = WETCOST
            rho_minus_C2[cover_C2==0] = WETCOST

            stego_C2 = embedding_simulator(cover_C2, rho_plus_C2, rho_minus_C2, message_length, index)
            ######################## STC embedding #############
            
            ############################ restore C2 part #############
            stego = cover.clone()
            stego = torch.flatten(stego, start_dim=2, end_dim=3)
            stego[:,:,idxC2] = stego_C2
            stego = stego.reshape(cover.shape)

            m = stego - cover
            changeRate += torch.sum(torch.abs(m))
            print(torch.mean(torch.abs(m)))

            final_m = stego - ori_cover
            m1 = cover - ori_cover
            # ############################ restore C2 part #############
            # message_length = args.imageSize*args.imageSize*args.payload
            # stego2 = embedding_simulator_img(ori_cover, rho_plus, rho_minus, message_length, index)

            # m = stego2 - ori_cover
            # stego = cover + m * maskC2
            # stego[stego == -1] = 0
            # stego[stego == 256] = 255

            # changeRate += torch.sum(torch.abs(m * maskC2))

            # print('m: %.4f m2: %.4f'%(torch.mean(torch.abs(m)).item(), torch.mean(torch.abs(m * maskC2)).item()))

            # final_m = stego - ori_cover

            # save images
            for k in range(0,stego.shape[0]):
                # cost_p1 = rho_plus[k,0].detach().cpu().numpy()
                # cost_p1[np.isinf(cost_p1)] = 10e+10
                # cost_m1 = rho_minus[k,0].detach().cpu().numpy()
                # cost_m1[np.isinf(cost_m1)] = 10e+10
                # sio.savemat( ('%s%s/%d_cost.mat'%(args.root, args.config, index[k])), mdict={'cost_p1': cost_p1, 'cost_m1': cost_m1})


                img = stego[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_stego.png' %(args.root, args.config, index[k]), img)

                # modify = (torch.round(m)+2)%2*255
                modify2 = (torch.round(m * maskC2)+1)*127
                modify2 = modify2[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_modify.png' %(args.root, args.config, index[k]), modify)

                modify1 = (torch.round(m1)+1)*127
                modify1 = modify1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()

                final_modify = (torch.round(final_m)+1)*127
                final_modify = final_modify[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                
                im = Image.fromarray(np.uint8(img[:,:,0]))
                im.save( ('%s%s/%d.pgm' %(args.root, args.config, index[k])))


                mod1 = Image.fromarray(np.uint8(modify1[:,:,0]))
                mod2 = Image.fromarray(np.uint8(modify2[:,:,0]))
                final_mod = Image.fromarray(np.uint8(final_modify[:,:,0]))

                imslst = [final_mod, mod1, mod2, im] 
                result = Image.new('RGBA', (args.imageSize * len(imslst), args.imageSize))
                for i, im in enumerate(imslst):
                    result.paste(im, box=(i * args.imageSize , 0))
                result.save(('%s%s/%d.png' %(args.root, args.config, index[k])))

    changeRate = changeRate / (10000.0 * args.imageSize * args.imageSize)
    with open('sim_embedding.txt','a') as f:
        f.write('%s: %.4f\n'%(args.config, changeRate.item()))

    print('Stegan down for {}'.format(args.datacover))
    print('Output path {}'.format(args.root + args.config))

if __name__ == '__main__':
    main()

