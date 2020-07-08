import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

print(torch.__version__[:3])
if(float(torch.__version__[:3])<1.4):
    print("requires pytorch 1.4 or higher")
    
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
import time
import os
import sys

#import MIND implementation from voxelmorph pull request https://github.com/voxelmorph/voxelmorph/pull/145
sys.path.append('voxelmorph/pytorch/')
import losses
print(losses.mind_loss)

#set empty arrays for images and mind features
H = 192; W = 160; D = 256;
imgs = torch.zeros(20,H,W,D)
mindssc = torch.zeros(20,12,H,W,D)

#load affinely pre-aligned "Beyond the Cranial Vault" training scans 1-10, 21-30 (31-40 are reserved for testing)
list_train = torch.cat((torch.arange(10),torch.arange(20,30)),0)+1
for i in range(20):
    img_fixed = torch.from_numpy(nib.load('/data/user/AbdomenPreAffine/Training/img/img00'+str(int(list_train[i])).zfill(2)+'.nii.gz').get_data()).float()
    imgs[i] = (img_fixed+1000)/500
    with torch.no_grad():
        mindssc[i] = losses.MINDSSC(imgs[i:i+1].unsqueeze(1).cuda(),3,3).cpu()
        
        
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice

    
avg5 = nn.AvgPool3d((5,5,5),stride=(1,1,1),padding=(2,2,2)).cuda()
o_m = H//3#H//3
o_n = W//3#W//3
o_o = D//3#D//3
corner = False

#strided grid for Obelisk features
print('numel_o',o_m*o_n*o_o)
ogrid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,o_m,o_n,o_o),align_corners=corner).view(1,1,-1,1,3).cuda()

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.xavier_normal(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.0)

def countParameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

#unchanged from public referenced code https://github.com/multimodallearning/pdd_net  
class OBELISK(nn.Module):
    def __init__(self):

        super(OBELISK, self).__init__()
        channels = 24#16
        self.offsets = nn.Parameter(torch.randn(2,channels*2,3)*0.05)
        self.layer0 = nn.Conv3d(1, 4, 5, stride=2, bias=False, padding=2)
        self.batch0 = nn.BatchNorm3d(4)

        self.layer1 = nn.Conv3d(channels*8, channels*4, 1, bias=False, groups=1)
        self.batch1 = nn.BatchNorm3d(channels*4)
        self.layer2 = nn.Conv3d(channels*4, channels*4, 3, bias=False, padding=1)
        self.batch2 = nn.BatchNorm3d(channels*4)
        self.layer3 = nn.Conv3d(channels*4, channels*1, 1)


    def forward(self, input_img):
        img_in = F.avg_pool3d(input_img,3,padding=1,stride=2)
        img_in = F.relu(self.batch0(self.layer0(img_in)))
        sampled = F.grid_sample(img_in,ogrid_xyz + self.offsets[0,:,:].view(1,-1,1,1,3),align_corners=corner).view(1,-1,o_m,o_n,o_o)
        sampled -= F.grid_sample(img_in,ogrid_xyz + self.offsets[1,:,:].view(1,-1,1,1,3),align_corners=corner).view(1,-1,o_m,o_n,o_o)

        x = F.relu(self.batch1(self.layer1(sampled)))
        x = F.relu(self.batch2(self.layer2(x)))
        features = self.layer3(x)
        return features



disp_range = 0.4 #q range of displacements (pytorch -1..+1)
displacement_width = 15 #number of steps per dimension
shift_xyz = F.affine_grid(disp_range*torch.eye(3,4).unsqueeze(0),(1,1,displacement_width,displacement_width,displacement_width),align_corners=corner).view(1,1,-1,1,3).cuda()

#_,_,H,W,D = img00.size()
grid_size = 29 #number of control points per dimension
grid_xyz = F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,grid_size,grid_size,grid_size),align_corners=corner).view(1,-1,1,1,3).cuda()


net = OBELISK()
print(countParameters(net))
#decomposed displacement label space
shift_x = shift_xyz.view(displacement_width,displacement_width,displacement_width,3)[(displacement_width-1)//2,:,:,:].reshape(1,1,-1,1,3)
shift_y = shift_xyz.view(displacement_width,displacement_width,displacement_width,3)[:,(displacement_width-1)//2,:,:].reshape(1,1,-1,1,3)
shift_z = shift_xyz.view(displacement_width,displacement_width,displacement_width,3)[:,:,(displacement_width-1)//2,:].reshape(1,1,-1,1,3)
shift_2d = torch.cat((shift_x,shift_y,shift_z),3)


class subplanar_pdd(nn.Module):
    def __init__(self):

        super(subplanar_pdd, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1,.1,1,1,.1,5]))#1]))#.cuda()

        self.pad1 = nn.ReplicationPad3d((0,0,2,2,2,2))#.cuda()
        self.avg1 = nn.AvgPool3d((3,3,1),stride=1)#.cuda()
        self.max1 = nn.MaxPool3d((3,3,1),stride=1)#.cuda()
        self.pad2 = nn.ReplicationPad3d((0,0,2,2,2,2))#.cuda()##


    def forward(self, feat00,feat50,shift_2d_min):

        #pdd correlation layer with 2.5D decomposition (slightly unrolled)
        pdd_cost = torch.zeros(1,grid_size**3,displacement_width,displacement_width,3).cuda()
        xyz8 = grid_size**2
        for i in range(grid_size): 
            moving_unfold = F.grid_sample(feat50,grid_xyz[:,i*xyz8:(i+1)*xyz8,:,:,:] + shift_2d_min[:,i*xyz8:(i+1)*xyz8,:,:,:],padding_mode='border',align_corners=corner)
            fixed_grid = F.grid_sample(feat00,grid_xyz[:,i*xyz8:(i+1)*xyz8,:,:,:],align_corners=corner)
            pdd_cost[:,i*xyz8:(i+1)*xyz8,:,:,:] = self.alpha[1]+self.alpha[0]*torch.sum(torch.pow(fixed_grid-moving_unfold,2),1).view(1,-1,displacement_width,displacement_width,3)

        pdd_cost = pdd_cost.view(1,-1,displacement_width,displacement_width,3)

        # approximate min convolution / displacement compatibility
        cost = (self.avg1(-self.max1(-self.pad1(pdd_cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,4,0,1).view(1,3*displacement_width**2,grid_size,grid_size,grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0,2,3,4,1).view(1,-1,displacement_width,displacement_width,3)

        # second path
        cost = self.alpha[4]+self.alpha[2]*pdd_cost+self.alpha[3]*cost_avg
        cost = (self.avg1(-self.max1(-self.pad1(cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2,3,4,0,1).view(1,3*displacement_width**2,grid_size,grid_size,grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0,2,3,4,1).view(grid_size**3,displacement_width**2,3)

        #probabilistic and continuous output
        cost_soft = F.softmax(-self.alpha[5]*cost_avg,1).view(-1,1,displacement_width,displacement_width,3)

        pred_xyz = 0.5*(cost_soft.view(-1,displacement_width**2,3,1)*shift_2d.view(1,displacement_width**2,3,3)).sum(1).sum(1)




        return cost_soft,pred_xyz,cost_avg
    

# GridNet and fit_sub2dense are used for instance optimisation (fitting of 2.5D displacement costs)
class GridNet(nn.Module):
    def __init__(self,grid_x,grid_y,grid_z):
        super(GridNet, self).__init__()
        self.params = nn.Parameter(0.1*torch.randn(1,3,grid_x,grid_y,grid_z))

    def forward(self):
        return self.params

smooth_hw2 = 3
H2 = H//3; W2 = W//3; D2 = D//3###

def fit_sub2dense(pred_xyz,grid_xyz,cost_avg,alpha,H,W,D,lambda_w=1.5,max_iter=100):

    cost2d = F.softmax(-alpha[5]*cost_avg,1).view(-1,1,displacement_width,displacement_width,3)
    
    with torch.enable_grad():
        net = GridNet(H2,W2,D2)
        net.params.data = pred_xyz.permute(0,4,1,2,3).detach()+torch.randn_like(pred_xyz.permute(0,4,1,2,3))*0.05
        net.cuda()
        avg5 = nn.AvgPool3d((3,3,3),stride=(1,1,1),padding=(1,1,1)).cuda()


        optimizer = optim.Adam(net.parameters(), lr=0.02)
        lambda_weight = lambda_w#1.5#5

        for iter in range(max_iter):
            optimizer.zero_grad()
            #second-order B-spline transformation model
            fitted_grid = (avg5(avg5(net())))
            #resampling transformation network to chosen control point spacing
            sampled_net = F.grid_sample(fitted_grid,grid_xyz,align_corners=True).permute(2,0,3,4,1)/disp_range
            #sampling the 2.5D displacement probabilities at 3D vectors
            sampled_cost = 0.33*F.grid_sample(cost2d[:,:,:,:,0],sampled_net[:,:,:,0,:2],align_corners=True)
            sampled_cost += 0.33*F.grid_sample(cost2d[:,:,:,:,1],sampled_net[:,:,:,0,torch.Tensor([0,2]).long()],align_corners=True)
            sampled_cost += 0.33*F.grid_sample(cost2d[:,:,:,:,2],sampled_net[:,:,:,0,1:],align_corners=True)
            #maximise probabilities
            loss = (-sampled_cost).mean()
            #minimise diffusion regularisation penalty
            reg_loss = lambda_weight*((fitted_grid[0,:,:,1:,:]-fitted_grid[0,:,:,:-1,:])**2).mean()+            lambda_weight*((fitted_grid[0,:,1:,:,:]-fitted_grid[0,:,:-1,:,:])**2).mean()+            lambda_weight*((fitted_grid[0,:,:,:,1:]-fitted_grid[0,:,:,:,:-1])**2).mean()
            
            (reg_loss+loss).backward()

            optimizer.step()
    #return both low-resolution and high-resolution transformation
    dense_flow_fit = F.interpolate(fitted_grid.detach(),size=(H,W,D),mode='trilinear',align_corners=True)

    return dense_flow_fit,fitted_grid

#data augmentation
def augmentAffine(img_in, mind_in, strength=0.05):
    """
    3D affine augmentation on image and segmentation mini-batch on GPU.
    (affine transf. is centered: trilinear interpolation and zero-padding used for sampling)
    :input: img_in batch (torch.cuda.FloatTensor), mind_in batch (torch.cuda.FloatTensor)
    :return: augmented BxCxTxHxW image batch (torch.cuda.FloatTensor), augmented BxTxHxW seg batch (torch.cuda.LongTensor)
    """
    B,C,D,H,W = img_in.size()
    affine_matrix = (torch.eye(3,4).unsqueeze(0) + torch.randn(B, 3, 4) * strength).to(img_in.device)

    meshgrid = F.affine_grid(affine_matrix,torch.Size((B,1,D,H,W)),align_corners=corner)

    img_out = F.grid_sample(img_in, meshgrid,padding_mode='border',align_corners=corner)
    mind_out = F.grid_sample(mind_in, meshgrid,padding_mode='border',align_corners=corner)

    return img_out, mind_out


# compute jacobian determinant as measure of deformation complexity
def jacobian_determinant_3d(dense_flow):
    B,_,H,W,D = dense_flow.size()
    
    dense_pix = dense_flow*(torch.Tensor([H-1,W-1,D-1])/2).view(1,3,1,1,1).to(dense_flow.device)
    gradz = nn.Conv3d(3,3,(3,1,1),padding=(1,0,0),bias=False,groups=3)
    gradz.weight.data[:,0,:,0,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradz.to(dense_flow.device)
    grady = nn.Conv3d(3,3,(1,3,1),padding=(0,1,0),bias=False,groups=3)
    grady.weight.data[:,0,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    grady.to(dense_flow.device)
    gradx = nn.Conv3d(3,3,(1,1,3),padding=(0,0,1),bias=False,groups=3)
    gradx.weight.data[:,0,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1)
    gradx.to(dense_flow.device)
    with torch.no_grad():
        jacobian = torch.cat((gradz(dense_pix),grady(dense_pix),gradx(dense_pix)),0)+torch.eye(3,3).view(3,3,1,1,1).to(dense_flow.device)
        jacobian = jacobian[:,:,2:-2,2:-2,2:-2]
        jac_det = jacobian[0,0,:,:,:]*(jacobian[1,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[1,2,:,:,:]*jacobian[2,1,:,:,:])-        jacobian[1,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[2,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[2,1,:,:,:])+        jacobian[2,0,:,:,:]*(jacobian[0,1,:,:,:]*jacobian[1,2,:,:,:]-jacobian[0,2,:,:,:]*jacobian[1,1,:,:,:])

    return jac_det


torch.cuda.empty_cache()
torch.cuda.reset_max_memory_cached()
torch.cuda.reset_max_memory_allocated()

#initialise trainable network parts
reg2d = subplanar_pdd()
reg2d.cuda()

net = OBELISK()
net.apply(init_weights)
net.cuda()
net.train()

#set-up 2D offsets for multi-step 2.5D estimation
shift_2d_min = shift_2d.repeat(1,grid_size**3,1,1,1)
shift_2d_min.requires_grad = False

#train using Adam with weight decay and exponential LR decay
optimizer = optim.AdamW(list(net.parameters())+list(reg2d.parameters()),lr=0.005)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

#some running metrics
run_mind = np.zeros(1001)
run_diff = np.zeros(1001)
run_dice = torch.zeros(0,13)
run_jac = torch.zeros(0,2)

idx_train = torch.cat((torch.arange(0,10),torch.arange(10,20)),0)

torch.cuda.synchronize()
t0 = time.time()
init_memory0 = torch.cuda.max_memory_allocated()
init_memory1 = torch.cuda.max_memory_cached()
sub_fit = torch.zeros(3,grid_size**3).cuda()
torch.cuda.synchronize()
t1 = time.time()
print('time','%0.3f'%(t1-t0),'sec. init alloc','%0.3f'%(init_memory0*1e-9),'GByte. init cached','%0.3f'%(init_memory1*1e-9))
#run for 1000 iterations / 250 epochs
for i in range(1001):

    #select random training pair (mini-batch=4 averaging at the end)
    idx = idx_train[torch.randperm(20)][:2]
    #fixed scan and MIND features are augmented
    img00, mind_aug = augmentAffine(imgs[idx[0:1]].unsqueeze(1).cuda(),,mindssc[idx[0:1],:,:,:].cuda(),0.0375)
    img50 = imgs[idx[1:2]].unsqueeze(1).cuda()
    
    #extract obelisk features with channels=24 and stride=3
    feat00 = net(img00) #00 is fixed
    feat50 = net(img50) #50 is moving

    #find initial through-plane offsets (without gradient tacking)
    with torch.no_grad():
        #run forward path with previous weights
        cost_soft2d,pred2d,cost_avg = reg2d(feat00.detach(),feat50.detach(),shift_2d.repeat(1,grid_size**3,1,1,1))
        pred2d = pred2d.view(1,grid_size,grid_size,grid_size,3)
        #perform instance fit
        dense_sub,sub_fit = fit_sub2dense(pred2d.detach(),grid_xyz.detach(),cost_avg.detach(),reg2d.alpha.detach(),H,W,D,5,30)
        #slighlty augment the found through-plane offsets
        sub_fit2 = sub_fit.view(3,-1) + 0.05*torch.randn(3,grid_size**3).cuda()
        shift_2d_min[0,:,:,0,2] = sub_fit2.view(3,-1)[2,:].view(-1,1).repeat(1,displacement_width**2)
        shift_2d_min[0,:,:,1,1] = sub_fit2.view(3,-1)[1,:].view(-1,1).repeat(1,displacement_width**2)
        shift_2d_min[0,:,:,2,0] = sub_fit2.view(3,-1)[0,:].view(-1,1).repeat(1,displacement_width**2)
        shift_2d_min.requires_grad = False

    #run 2.5D probabilistic dense displacement (pdd2.5-net)
    cost_soft2d,pred2d,cost_avg = reg2d(feat00,feat50,shift_2d_min)
    
    #warm-up phase with stronger regularisation
    if(i<100):
        lambda_weight_2d = float(torch.linspace(0.75,0.025,100)[i])
    else:
        lambda_weight_2d = 0.025
    pred2d = pred2d.view(1,grid_size,grid_size,grid_size,3)
    #diffusion regularisation loss
    diffloss = lambda_weight_2d*((pred2d[0,:,1:,:,:]-pred2d[0,:,:-1,:,:])**2).mean()+\
            lambda_weight_2d*((pred2d[0,1:,:,:,:]-pred2d[0,:-1,:,:,:])**2).mean()+\
            lambda_weight_2d*((pred2d[0,:,:,1:,:]-pred2d[0,:,:,:-1,:])**2).mean()

    #nonlocal MIND loss
    fixed_mind = F.grid_sample(mind_aug.cuda(),grid_xyz,padding_mode='border',align_corners=corner).detach()#.long().squeeze(1)
    moving_unfold = F.grid_sample(mindssc[idx[1:2],:,:,:].cuda(),grid_xyz + shift_2d_min,padding_mode='border',align_corners=corner)
    nonlocal_mind = 1/3*torch.sum(moving_unfold*cost_soft2d.view(1,1,-1,displacement_width**2,3),[3,4]).view(1,12,grid_size**3,1,1)
    mindloss2d = ((nonlocal_mind-fixed_mind)**2)#*class_weight.view(1,-1,1,1,1)
    mindloss = mindloss2d.mean()

    run_diff[i] = diffloss.item()
    run_mind[i] = mindloss.item()
    (diffloss+mindloss).backward()

    #implicit mini-batch of 4 (and LR-decay)
    if(i%4==0):
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    #verbose ON: report some numbers and run inference on (potentially unseen test images)
    if(i%5==0):
        print(i,time.time()-t1,'mind',mindloss.item(),'diff',diffloss.item())
        print(reg2d.alpha)
        with torch.no_grad():
            feat00 = net(imgs[0:1].unsqueeze(1).cuda())#net(img00)# #00 is fixed
            feat50 = net(imgs[1:2].unsqueeze(1).cuda())#net(img50)# #50 is moving
            cost_soft2d,pred2d,cost_avg = reg2d(feat00,feat50,shift_2d.repeat(1,grid_size**3,1,1,1))
            pred2d = pred2d.view(1,grid_size,grid_size,grid_size,3)

            #instance based optimisation / fitting of 2.5D displacement cost
            dense_sub,sub_fit = fit_sub2dense(pred2d.detach(),grid_xyz.detach(),cost_avg.detach(),reg2d.alpha.detach(),H,W,D,5,30)
            identity = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,H,W,D),align_corners=True)

            #second refinement step (see Figure 1 right in paper)
            shift_2d_min[0,:,:,0,2] = sub_fit.view(3,-1)[2,:].view(-1,1).repeat(1,displacement_width**2)
            shift_2d_min[0,:,:,1,1] = sub_fit.view(3,-1)[1,:].view(-1,1).repeat(1,displacement_width**2)
            shift_2d_min[0,:,:,2,0] = sub_fit.view(3,-1)[0,:].view(-1,1).repeat(1,displacement_width**2)
            shift_2d_min.requires_grad = False

            #new dissimilarity planes are computed based on previous fit to reduce approximation error 
            cost_soft2d,pred2d,cost_avg = reg2d(feat00.detach(),feat50.detach(),shift_2d_min)
            pred2d = pred2d.view(1,grid_size,grid_size,grid_size,3)

            #instance based optimisation / fitting of 2.5D displacement cost is repeated
            dense_sub,sub_fit = fit_sub2dense(pred2d.detach(),grid_xyz.detach(),cost_avg.detach(),reg2d.alpha.detach(),H,W,D,10,30)

            #if segmentations are available for some validation/training data, Dice can be computed 
            #seg_w2d = F.grid_sample(segs[1:2,:,:,:].float().unsqueeze(1).cuda(),identity+dense_sub.permute(0,2,3,4,1),mode='nearest',padding_mode='border',align_corners=True).detach().long().squeeze()
            #d2d = dice_coeff(segs[0:1,:,:,:].cuda(),seg_w2d.cuda(),14).cpu()
            #print(d2d,d2d.mean())
            #run_dice = torch.cat((run_dice,d2d.view(1,-1)),0)

            #complexity of transformation and foldings
            jacdet = jacobian_determinant_3d(dense_sub)
            jac2 = torch.Tensor([torch.std(jacdet.view(-1)),torch.mean((jacdet.view(-1)<0).float())]).cpu()
            print(torch.std(jacdet.view(-1)),torch.mean((jacdet.view(-1)<0).float()))
            run_jac = torch.cat((run_jac,jac2.view(1,-1)),0)

init_memory0 = torch.cuda.max_memory_allocated()
init_memory1 = torch.cuda.max_memory_cached()

torch.cuda.synchronize()
t2 = time.time()

print('time','%0.3f'%(t2-t1),'sec. back alloc','%0.3f'%(init_memory0*1e-9),' GByte. back cached','%0.3f'%(init_memory1*1e-9))
