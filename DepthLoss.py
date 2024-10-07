import torch
import torch.nn as nn


#loss 10 depth with sigmoid
class DepthLossF(nn.Module):
    def __init__(self,max_distence_from_mean = 0.3,min_distence_from_mean = 0.03,max_offset = 0.04):
        super(DepthLossF, self).__init__()
        self.max_distence_from_mean = max_distence_from_mean
        self.min_distence_from_mean = min_distence_from_mean
        self.max_offset = max_offset
        
    def forward(self, depth_back, depth_front,offet_back,offet_front):
        with torch.no_grad():
            median = torch.unique(torch.cat((depth_back, depth_front)).flatten())
        median = torch.mean(median)
        
        loss_back1 = torch.relu((median) - depth_back)
        loss_back2 = torch.relu(depth_back - (median+self.max_distence_from_mean))
        loss_front1 = torch.relu(depth_front - (median))
        loss_front2 = torch.relu((median-self.max_distence_from_mean) - depth_front)

        loss_offset_back1 = torch.relu(offet_back - self.max_offset)
        loss_offset_front1= torch.relu(offet_front - self.max_offset)
        loss_offset_back2 = torch.relu(offet_back + self.max_offset)
        loss_offset_front2 = torch.relu(offet_front + self.max_offset)
        
        loss = torch.mean(loss_back1 + loss_back2 ) + torch.mean(loss_front1 + loss_front2) + torch.mean(loss_offset_back1 + loss_offset_front1 + loss_offset_back2 + loss_offset_front2)
        return loss
    

'''
#loss 9 depth with sigmoid
class DepthLossF(nn.Module):
    def __init__(self,max_distence_from_mean = 1,min_distence_from_mean = 0.05,max_offset = 0.2):
        super(DepthLossF, self).__init__()
        self.max_distence_from_mean = max_distence_from_mean
        self.min_distence_from_mean = min_distence_from_mean
        self.max_offset = max_offset
        
    def forward(self, depth_back, depth_front,offet_back,offet_front):
        with torch.no_grad():
            median = torch.unique(torch.cat((depth_back, depth_front)).flatten())
        median = torch.median(median)
        
        loss_back1 = torch.relu((median) - depth_back)
        loss_back2 = torch.relu(depth_back - (median+self.max_distence_from_mean))
        loss_front1 = torch.relu(depth_front - (median))
        loss_front2 = torch.relu((median-self.max_distence_from_mean) - depth_front)
        loss_offset_back1 = torch.relu(offet_back - self.max_offset)
        loss_offset_front1= torch.relu(offet_front - self.max_offset)
        loss_offset_back2 = torch.relu(offet_back + self.max_offset)
        loss_offset_front2 = torch.relu(offet_front + self.max_offset)

    
        loss = torch.mean(loss_back1 + loss_back2 + loss_front1 + loss_front2) + torch.mean(loss_offset_back1 + loss_offset_front1 + loss_offset_back2 + loss_offset_front2)
        return loss
    
#loss 8 with out normalization
class DepthLossF(nn.Module):
    def __init__(self,max_distence_from_mean = 1,min_distence_from_mean = 0.05,max_offset = 0.2):
        super(DepthLossF, self).__init__()
        self.max_distence_from_mean = max_distence_from_mean
        self.min_distence_from_mean = min_distence_from_mean
        self.max_offset = max_offset
        
    def forward(self, depth_back, depth_front,):
        with torch.no_grad():
            median = torch.unique(torch.cat((depth_back, depth_front)).flatten())
        median = torch.median(median)
        
        loss_back1 = torch.relu((median) - depth_back)
        loss_back2 = torch.relu(depth_back - (median+self.max_distence_from_mean))
        loss_front1 = torch.relu(depth_front - (median))
        loss_front2 = torch.relu((median-self.max_distence_from_mean) - depth_front)
    
        loss = torch.mean(loss_back1 + loss_back2 + loss_front1 + loss_front2)
        return loss
'''
    
'''
#loss 7
class DepthLossF(nn.Module):
    def __init__(self,min_distence_from_mean = 0.05):
        super(DepthLossF, self).__init__()
        self.min_distence_from_mean = min_distence_from_mean
        
    def forward(self, depth_back, depth_front):
        with torch.no_grad():
            median = torch.unique(torch.cat((depth_back, depth_front)).flatten())
        median = torch.mean(median)
        
        loss_back = torch.relu((median+self.min_distence_from_mean) - depth_back)
        loss_front = torch.relu(depth_front - (median-self.min_distence_from_mean))
        loss_above_0_front = torch.relu(-depth_front-0.001)
        loss_above_0_back = torch.relu(-depth_back-0.001)
        loss = torch.mean(0.4*loss_back + 0.4*loss_front+0.1*loss_above_0_front+0.1*loss_above_0_back)
        return loss
'''
'''
#loss6
class DepthLossF(nn.Module):
    def __init__(self):
        """
        lower_margin: Minimum distance between back and front.
        upper_margin: Maximum allowed distance between back and front.
        """
        super(DepthLossF, self).__init__()
        
    def forward(self, depth_back, depth_front):
        with torch.no_grad():
            median = torch.unique(torch.cat((depth_back, depth_front)).flatten())
        median = torch.median(median)
        
        loss_back = torch.clamp(median - depth_back, min=0.0)
        loss_front = torch.clamp(depth_front - median,min=0.0)
        loss = torch.mean(loss_back + loss_front)
        return loss
'''
"""
#loss5
class DepthLossF(nn.Module):
    def __init__(self, margin=0.75):
        super(DepthLossF, self).__init__()
        self.margin = margin  # You can set this to control how much distance is desired.

    def forward(self, x1, x2):
        # Calculate Euclidean distance between two vectors x1 and x2
        distance = torch.norm(x1 - x2, p=2, dim=-1)

        # The loss increases as the distance decreases below a certain margin
        loss = torch.clamp(self.margin - distance, min=0.0)  # loss is zero when distance >= margin

        return loss.mean()
"""
'''
#loss4
class DepthLossF(nn.Module):
    def __init__(self, C):
        super(DepthLossF, self).__init__()
        self.Min_diffrence = C
    
    def forward(self, back_depth, front_depth):

        diff = torch.mean(back_depth) - torch.mean(front_depth)

        loss1 = torch.relu(self.Min_diffrence - diff)
        
        #loss2 = torch.relu(diff - 2 * self.Min_diffrence)

        return loss1
'''
'''
class DepthLossF3(nn.Module):
    def __init__(self, C):
        super(DepthLossF, self).__init__()
        self.Min_diffrence = C
    
    def forward(self, back_depth, front_depth):

        diff = back_depth - front_depth

        loss1 = torch.relu(self.Min_diffrence - diff)
        #loss2 = torch.relu(diff - 2 * self.Min_diffrence)

        loss = torch.mean(loss1)#+ loss2)
        return loss
'''
'''
class DepthLossF2(nn.Module):
    def __init__(self, C):
        super(DepthLossF, self).__init__()
        self.Min_diffrence = C
    
    def forward(self, D1, D2):

        diff = D1 - D2

        loss1 = torch.relu(self.Min_diffrence - diff)
        loss2 = torch.relu(diff - 2 * self.Min_diffrence)

        loss = torch.mean(loss1 + loss2)
        return loss
'''
'''
class DepthLossF1(nn.Module):
    def __init__(self, C):
        super(DepthLossF1, self).__init__()
        self.Diffrence_between_means = C
    
    def forward(self, D1, D2):
        diffrence = torch.abs(D1 - D2)
        loss = torch.mean(torch.relu(diffrence-self.Diffrence_between_means))
        return loss
'''