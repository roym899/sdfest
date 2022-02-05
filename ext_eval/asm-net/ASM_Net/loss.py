import numpy as np
from numpy import linalg as LA
import copy
import torch
import torch.nn as nn
# from pytorch3d.loss import chamfer_distance


def quaternion2rotationPT( q ):
    """ Convert unit quaternion to rotation matrix
    
    Args:
        q(torch.tensor): unit quaternion (N,4)
    Returns:
        torch.tensor: rotation matrix (N,3,3)
    """
    r11 = (q[:,0]**2+q[:,1]**2-q[:,2]**2-q[:,3]**2).unsqueeze(0).T
    r12 = (2.0*(q[:,1]*q[:,2]-q[:,0]*q[:,3])).unsqueeze(0).T
    r13 = (2.0*(q[:,1]*q[:,3]+q[:,0]*q[:,2])).unsqueeze(0).T

    r21 = (2.0*(q[:,1]*q[:,2]+q[:,0]*q[:,3])).unsqueeze(0).T
    r22 = (q[:,0]**2+q[:,2]**2-q[:,1]**2-q[:,3]**2).unsqueeze(0).T
    r23 = (2.0*(q[:,2]*q[:,3]-q[:,0]*q[:,1])).unsqueeze(0).T

    r31 = (2.0*(q[:,1]*q[:,3]-q[:,0]*q[:,2])).unsqueeze(0).T
    r32 = (2.0*(q[:,2]*q[:,3]+q[:,0]*q[:,1])).unsqueeze(0).T
    r33 = (q[:,0]**2+q[:,3]**2-q[:,1]**2-q[:,2]**2).unsqueeze(0).T
    
    r = torch.cat( (r11,r12,r13,
                r21,r22,r23,
                r31,r32,r33), 1 )
    r = torch.reshape( r, (q.shape[0],3,3))
    
    return r

class ASMdeformationPT():
    def __init__( self, asm_info ):
        
        self.mean_shape = np.array( asm_info['mean_shape'], np.float32) # load mean shape
        self.component = np.array(asm_info['components' ], np.float32) # load components
        self.mean = np.array(asm_info['size_mean'], np.float32) # size mean
        self.std = np.array(asm_info['size_std'], np.float32) # size std
        # to tensor
        self.mean_shape = torch.from_numpy(self.mean_shape).cuda()
        self.component = torch.from_numpy(self.component).cuda()
        self.mean = torch.from_numpy(self.mean).cuda()
        self.std = torch.from_numpy(self.std).cuda()
    
    def deformation( self, dp ):
        """
        Deformation
        """
        bs = dp.shape[0]
        bs_deformed = torch.zeros((bs,self.component.shape[1]))
        for i in range(bs):
            deformed = copy.deepcopy( self.mean_shape )
            for c,p in zip( self.component, dp[i]):
                deformed += c*p
        
            deformed = (deformed * self.std)+self.mean
            bs_deformed[i] = deformed
            
        return bs_deformed.cuda()
    
class ReconstructionLoss(nn.Module):

    def __init__( self, asmd_pred, asmd_gt, mode=None ):
        super(ReconstructionLoss, self).__init__()
        self.asmd_pred = asmd_pred
        self.asmd_gt = asmd_gt
        self.mode = mode
        self.gt_shape = None
        self.pred_shape = None
        
    def forward(self, pred, gt ):
        
        pred_dp_param = pred[:,:-1]
        pred_scaling_param = pred[:,-1]
        
        gt_dp_param = gt[:,:-1]
        gt_scaling_param = gt[:,-1]
        pred_shape = pred_scaling_param * self.asmd_pred.deformation( pred_dp_param ).T
        gt_shape = gt_scaling_param * self.asmd_gt.deformation( gt_dp_param ).T
        pred_shape = pred_shape.T
        gt_shape = gt_shape.T
        pred_shape = pred_shape.reshape(pred_shape.shape[0],-1,3)
        gt_shape = gt_shape.reshape(gt_shape.shape[0],-1,3)
        
        #diff = gt_shape - pred_shape
        output, _ = chamfer_distance(gt_shape,pred_shape)
        
        if self.mode != None:
            self.gt_shape = gt_shape
            self.pred_shape = pred_shape
        return output


    
