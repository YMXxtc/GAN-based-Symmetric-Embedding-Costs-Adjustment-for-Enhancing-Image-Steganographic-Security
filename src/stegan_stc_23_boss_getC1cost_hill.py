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

# from ResUNet1DB import UNet
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,1,0'

WETCOST = 10e+10

def myParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='BB-cover-resample-256-C1stc-hill', help="train result")

    parser.add_argument('--datacover', help='path to dataset',default='/data1/ymx/BOSSBase/BB-cover-resample-256-testset/')
    parser.add_argument('--costroot', help='path to cost',default='/data1/ymx/HILL_bb/0.4/')
    parser.add_argument('--indexpath', help='path to index',default='./index_list/boss_gan_train_1w.npy')
    
    parser.add_argument('--root', default='/data1/ymx/', help="path to save result")

    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    
    parser.add_argument('--TS_PARA', type=int, default=1000000, help='parameter for double tanh')
    parser.add_argument('--payload', type=float, default=0.4, help='embeding rate')
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



class Dataset(data.Dataset):
    def __init__(self, cover_path, cost_path, index_path, transforms = None):
        self.index_list = np.load(index_path)
        self.cover_path = cover_path+'{}.pgm'

        self.cost_plus_path = cost_path+'{}_rhoP.mat'
        self.cost_minus_path = cost_path+'{}_rhoM.mat'

        self.transforms = transforms        
        

    def __getitem__(self,index):
        file_index = self.index_list[index]
        cover_path = self.cover_path.format(file_index)
        cover = cv2.imread(cover_path, -1)
        label = np.array([0, 1], dtype='int32')

        cost_plus_path = self.cost_plus_path.format(file_index)
        cost_minus_path = self.cost_minus_path.format(file_index)
        rho_plus = sio.loadmat(cost_plus_path)['costP']
        rho_minus = sio.loadmat(cost_minus_path)['costM']
        
        rand_ud = np.random.rand(256, 256)
        sample = {'cover':cover, 'rand_ud':rand_ud, 'label':label, 'index':file_index, 'rho_plus':rho_plus, 'rho_minus':rho_minus}
        
        if self.transforms:
            sample = self.transforms(sample)
            
        return sample

        
    def __len__(self):
        return len(self.index_list)


class ToTensor():
  def __call__(self, sample):
    cover, rand_ud, label, index, rho_plus, rho_minus  = sample['cover'], sample['rand_ud'], sample['label'], sample['index'], sample['rho_plus'], sample['rho_minus']

    cover = np.expand_dims(cover, axis=0)
    cover = cover.astype(np.float32)
    
    rand_ud = rand_ud.astype(np.float32)
    rand_ud = np.expand_dims(rand_ud,axis = 0)
    
    rho_plus = np.expand_dims(rho_plus, axis=0)
    rho_plus = rho_plus.astype(np.float32)
    rho_minus = np.expand_dims(rho_minus, axis=0)
    rho_minus = rho_minus.astype(np.float32)

    new_sample = {
      'cover': torch.from_numpy(cover),
      'rand_ud': torch.from_numpy(rand_ud),
      'label': torch.from_numpy(label).long(),
      'index':index,
      'rho_plus':rho_plus,
      'rho_minus':rho_minus
    }

    return new_sample


# # custom weights initialization called on netG and netD
# def weights_init_g(net):
#     for m in net.modules():
#         if isinstance(m,nn.Conv2d) and m.weight.requires_grad:
#             m.weight.data.normal_(0., 0.02)
#         elif isinstance(m, nn.ConvTranspose2d):
#             m.weight.data.normal_(0., 0.02)
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0., 0.02)
#             m.bias.data.fill_(0)
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.normal_(1.0, 0.02)
#             m.bias.data.fill_(0)


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
        args.manualSeed = random.randint(1, 10000) #10000
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    cudnn.benchmark = True
        
    
    transform = transforms.Compose([ToTensor(),])
    dataset = Dataset(args.datacover, args.costroot, args.indexpath, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                            shuffle=True, num_workers=int(args.workers),drop_last = True)


    ############### randomly divid ###############
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
    maskC1 = torch.from_numpy(maskC1).cuda()
    maskC2 = maskC2.reshape(args.imageSize,args.imageSize)
    maskC2 = torch.from_numpy(maskC2).cuda()
    ############### randomly divid ###############

    # adjust_bn_stats(netG, train_loader)

    changeRate = 0

    # netG.eval()
    with torch.no_grad():
        for i,sample in enumerate(dataloader,0):

            cover, rand_ud, label, index, rho_plus, rho_minus = sample['cover'], sample['rand_ud'], sample['label'], sample['index'], sample['rho_plus'], sample['rho_minus']
            cover, n, label, rho_plus, rho_minus = cover.cuda(), rand_ud.cuda(), label.cuda(), rho_plus.cuda(), rho_minus.cuda()

            ########################## extract C1 part ###########
            rho_plus_C1 = torch.flatten(rho_plus, start_dim=2, end_dim=3)[:,:,idxC1]
            rho_minus_C1 = torch.flatten(rho_minus, start_dim=2, end_dim=3)[:,:,idxC1]
            cover_C1 = torch.flatten(cover, start_dim=2, end_dim=3)[:,:,idxC1]
            # print(cover_C1.shape)
            ########################## extract C1 part ###########

            ######################## STC embedding #############
            message_length = args.imageSize*args.imageSize*args.payload / 2.0

            rho_plus_C1[rho_plus_C1 > WETCOST] = WETCOST
            rho_plus_C1[torch.isinf(rho_plus_C1)] = WETCOST
            rho_minus_C1[rho_minus_C1 > WETCOST] = WETCOST
            rho_minus_C1[torch.isinf(rho_minus_C1)] = 10e+10

            rho_plus_C1[cover_C1==255] = WETCOST
            rho_minus_C1[cover_C1==0] = WETCOST

            stego_C1 = embedding_simulator(cover_C1, rho_plus_C1, rho_minus_C1, message_length, index)
            ######################## STC embedding #############
            
            ############################ restore C1 part #############
            stego = cover.clone()
            stego = torch.flatten(stego, start_dim=2, end_dim=3)
            stego[:,:,idxC1] = stego_C1
            stego = stego.reshape(cover.shape)

            m = stego - cover
            changeRate += torch.sum(torch.abs(m))
            print(torch.mean(torch.abs(m)))
            ############################ restore C1 part #############

            stego[stego == -1] = 0
            stego[stego == 256] = 255

            # save images
            for k in range(0,stego.shape[0]):
                # cost_p1 = rho_plus[k,0].detach().cpu().numpy()
                # cost_p1[np.isinf(cost_p1)] = 10e+10
                # cost_m1 = rho_minus[k,0].detach().cpu().numpy()
                # cost_m1[np.isinf(cost_m1)] = 10e+10
                # sio.savemat( ('%s%s/%d_cost.mat'%(args.root, args.config, index[k])), mdict={'cost_p1': cost_p1, 'cost_m1': cost_m1})


                img = stego[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_stego.png' %(args.root, args.config, index[k]), img)

                modify = (torch.round(m)+2)%2*255
                # modify = (torch.round(m)+1)*127
                modify = modify[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_modify.png' %(args.root, args.config, index[k]), modify)

                # pro_m1 = p_minus*255
                # probas_m1 = pro_m1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                # #cv2.imwrite('%s%s/%d_probas.png' %(args.root,  args.config, index[k]), probas_m1)

                # pro_p1 = p_plus*255
                # probas_p1 = pro_p1[k,].detach().cpu().permute(1,2,0).type(torch.uint8).numpy()
                #cv2.imwrite('%s%s/%d_probas.png' %(args.root,  args.config, index[k]), probas_p1)
                
                # im = Image.fromarray(np.uint8(img[:,:,0]))
                # im.save( ('%s%s/%d.pgm' %(args.root, args.config, index[k])))


                mod = Image.fromarray(np.uint8(modify[:,:,0]))
                # p_m1 = Image.fromarray(np.uint8(probas_m1[:,:,0]))
                # p_p1 = Image.fromarray(np.uint8(probas_p1[:,:,0]))

                imslst = [mod, im]  #p_p1, p_m1, 
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