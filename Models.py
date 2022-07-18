import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from NetUtils import *
from FlowModel import FlowNet, FlowNetMulti, FlowNetMulti_compact

class Backward_warp(nn.Module):

    def __init__(self):
        super(Backward_warp,self).__init__()


    def _meshgrid(self,height,width):

        y_t = torch.linspace(0,height - 1, height).reshape(height,1) * torch.ones(1,width)
        x_t = torch.ones(height,1) * torch.linspace(0, width - 1, width).reshape(1,width)

        x_t_flat = x_t.reshape(1,1,height,width)
        y_t_flat = y_t.reshape(1,1,height,width)

        grid = torch.cat((x_t_flat,y_t_flat),1)

        return grid


    def _interpolate(self,img , x, y , out_height, out_width):

        num_batch,height,width,num_channel = img.size()
        height_f = float(height)
        width_f = float(width)

        x = torch.clamp(x,0,width - 1)
        y = torch.clamp(y,0,height - 1)

        x0_f = x.floor()
        y0_f = y.floor()
        x1_f = x0_f + 1.0
        y1_f = y0_f + 1.0

        x0 = torch.tensor(x0_f, dtype = torch.int64)
        y0 = torch.tensor(y0_f, dtype = torch.int64)
        x1 = torch.tensor(torch.clamp(x1_f, 0, width_f -1), dtype = torch.int64)
        y1 = torch.tensor(torch.clamp(y1_f, 0, height_f -1), dtype = torch.int64)
 
        dim1 = width * height
        dim2 = width
        base = torch.tensor((torch.arange(num_batch) * dim1),dtype = torch.int64).cuda()
        base = base.reshape(num_batch,1).repeat(1,out_height * out_width).view(-1)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        img_flat = img.reshape(-1,num_channel)

        Ia = img_flat[idx_a]
        Ib = img_flat[idx_b]
        Ic = img_flat[idx_c]
        Id = img_flat[idx_d]

        wa = ((x1_f-x) * (y1_f-y)).reshape(-1,1)
        wb = ((x1_f-x) * (y-y0_f)).reshape(-1,1)
        wc = ((x-x0_f) * (y1_f-y)).reshape(-1,1)
        wd = ((x-x0_f) * (y-y0_f)).reshape(-1,1)
        output = wa * Ia + wb * Ib + wc * Ic + wd *Id

        return output

    def _transform_flow(self,flow,input,downsample_factor):

        num_batch,num_channel,height,width = input.size()

        out_height = height
        out_width = width
        grid = self._meshgrid(height, width)
        if num_batch > 1:
            grid = grid.repeat(num_batch,1,1,1)

        control_point = grid.cuda() + flow
        input_t = input.permute(0,2,3,1)

        x_s_flat = control_point[:,0,:,:].contiguous().view(-1)
        y_s_flat = control_point[:,1,:,:].contiguous().view(-1)

        input_transformed = self._interpolate(input_t,x_s_flat,y_s_flat,out_height,out_width)

        input_transformed = input_transformed.reshape(num_batch,out_height,out_width,num_channel)

        output = input_transformed.permute(0,3,1,2)

        return output

    def forward(self,input,flow,downsample_factor = 1):

        return self._transform_flow(flow,input, downsample_factor)


class Backward_warp_predefined(nn.Module):

    def __init__(self,height,width):
        super(Backward_warp_predefined,self).__init__()
        self.meshgrid = self._meshgrid(height, width).cuda()

    def _meshgrid(self,height,width):

        y_t = torch.linspace(0,height - 1, height).reshape(height,1) * torch.ones(1,width)
        x_t = torch.ones(height,1) * torch.linspace(0, width - 1, width).reshape(1,width)

        x_t_flat = x_t.reshape(1,1,height,width)
        y_t_flat = y_t.reshape(1,1,height,width)

        grid = torch.cat((x_t_flat,y_t_flat),1)

        return grid


    def _interpolate(self,img , x, y , out_height, out_width):

        num_batch,height,width,num_channel = img.size()
        height_f = float(height)
        width_f = float(width)

        x = torch.clamp(x,0,width - 1)
        y = torch.clamp(y,0,height - 1)

        x0_f = x.floor()
        y0_f = y.floor()
        x1_f = x0_f + 1.0
        y1_f = y0_f + 1.0

        x0 = torch.tensor(x0_f, dtype = torch.int64)
        y0 = torch.tensor(y0_f, dtype = torch.int64)
        x1 = torch.tensor(torch.clamp(x1_f, 0, width_f -1), dtype = torch.int64)
        y1 = torch.tensor(torch.clamp(y1_f, 0, height_f -1), dtype = torch.int64)
 
        dim1 = width * height
        dim2 = width
        base = torch.tensor((torch.arange(num_batch) * dim1),dtype = torch.int64)
        base = base.reshape(num_batch,1).repeat(1,out_height * out_width).view(-1)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        img_flat = img.reshape(-1,num_channel)

        Ia = img_flat[idx_a]
        Ib = img_flat[idx_b]
        Ic = img_flat[idx_c]
        Id = img_flat[idx_d]

        wa = ((x1_f-x) * (y1_f-y)).reshape(-1,1)
        wb = ((x1_f-x) * (y-y0_f)).reshape(-1,1)
        wc = ((x-x0_f) * (y1_f-y)).reshape(-1,1)
        wd = ((x-x0_f) * (y-y0_f)).reshape(-1,1)
        output = wa * Ia + wb * Ib + wc * Ic + wd *Id

        return output

    def _transform_flow(self,flow,input,downsample_factor):

        num_batch,num_channel,height,width = input.size()

        out_height = height
        out_width = width

        control_point = self.meshgrid.repeat(num_batch,1,1,1) + flow
        input_t = input.permute(0,2,3,1)

        x_s_flat = control_point[:,0,:,:].contiguous().view(-1)
        y_s_flat = control_point[:,1,:,:].contiguous().view(-1)

        input_transformed = self._interpolate(input_t,x_s_flat,y_s_flat,out_height,out_width)

        input_transformed = input_transformed.reshape(num_batch,out_height,out_width,num_channel)

        output = input_transformed.permute(0,3,1,2)

        return output

    def forward(self,input,flow,downsample_factor = 1):

        return self._transform_flow(flow,input, downsample_factor)

class Backward_warp_multi(nn.Module):

    def __init__(self, height, width):
        super(Backward_warp_multi,self).__init__()

    def _meshgrid(self,height,width, K=1):
        y_t = torch.linspace(0,height - 1, height).reshape(height,1) * torch.ones(1,width)
        x_t = torch.ones(height,1) * torch.linspace(0, width - 1, width).reshape(1,width)

        x_t_flat = x_t.reshape(1,1,height,width)
        y_t_flat = y_t.reshape(1,1,height,width)

        grid = torch.cat((x_t_flat,y_t_flat),1)

        return grid.repeat(1,K,1,1)

    def _interpolate(self,img , x, y , out_height, out_width, K=1):
        num_batch,height,width,num_channel = img.size()
        height_f = float(height)
        width_f = float(width)

        x = torch.clamp(x,0,width - 1)
        y = torch.clamp(y,0,height - 1)

        x0_f = x.floor()
        y0_f = y.floor()
        x1_f = x0_f + 1.0
        y1_f = y0_f + 1.0

        x0 = torch.tensor(x0_f, dtype = torch.int64)
        y0 = torch.tensor(y0_f, dtype = torch.int64)
        x1 = torch.tensor(torch.clamp(x1_f, 0, width_f -1), dtype = torch.int64)
        y1 = torch.tensor(torch.clamp(y1_f, 0, height_f -1), dtype = torch.int64)
 
        dim1 = width * height
        dim2 = width
        base = torch.tensor((torch.arange(num_batch) * dim1),dtype = torch.int64)
        base = base.reshape(num_batch,1).repeat(1,out_height * out_width * K).view(-1)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        img_flat = img.reshape(-1,num_channel)

        Ia = img_flat[idx_a]
        Ib = img_flat[idx_b]
        Ic = img_flat[idx_c]
        Id = img_flat[idx_d]

        wa = ((x1_f-x) * (y1_f-y)).reshape(-1,1)
        wb = ((x1_f-x) * (y-y0_f)).reshape(-1,1)
        wc = ((x-x0_f) * (y1_f-y)).reshape(-1,1)
        wd = ((x-x0_f) * (y-y0_f)).reshape(-1,1)
        output = wa * Ia + wb * Ib + wc * Ic + wd *Id

        return output

    def _transform_flow(self,flow,attention,input,downsample_factor):
        num_batch,num_channel,height,width = input.size()
        K = attention.size(1)

        out_height = height
        out_width = width
        grid = self._meshgrid(height, width, K = K)

        if num_batch > 1:
            grid = grid.repeat(num_batch,1,1,1)

        control_point = grid.cuda() + flow

        input_t = input.permute(0,2,3,1)

        for k in range(K):
            x_s_flat = control_point[:,0::2,:,:].contiguous().view(-1)
            y_s_flat = control_point[:,1::2,:,:].contiguous().view(-1)
            att_flat = attention[:,:,:,:].contiguous().view(-1)

            input_transformed =  self._interpolate(input_t,x_s_flat,y_s_flat,out_height,out_width,K=K)
            input_transformed = torch.mul(input_transformed, att_flat.unsqueeze(1)).view(num_batch,K,height,width,num_channel)
            output = input_transformed.sum(dim=1).permute(0,3,1,2)

        return output

    def forward(self,input,flow,attention,downsample_factor = 1):
        return self._transform_flow(flow,attention,input, downsample_factor)

class FlowNet_MultiscaleWarp_2Frame(nn.Module):
    ''' 2-frame baseline '''
    def __init__(self):
        super(FlowNet_MultiscaleWarp_2Frame, self).__init__()
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, img0, img1, img2):
        ## encode img1
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(img1)

        # flow img1->img0    (from target to source)
        flow_10 = self.FlowNet(img1, img0)
        flow_10_1, flow_10_2, flow_10_3, flow_10_4 = flow_10['flow_1'],  flow_10['flow_2'],  flow_10['flow_3'], flow_10['flow_4']
        img0_conv1, img0_conv2, img0_conv3, img0_conv4 = self.Encoder(img0)
        warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4 = (self.Backward_warp(img0_conv1, flow_10_1),
                                                                self.Backward_warp(img0_conv2, flow_10_2),
                                                                self.Backward_warp(img0_conv3, flow_10_3),
                                                                self.Backward_warp(img0_conv4, flow_10_4))
        # fusion
        sythsis_output  = self.UNet_decoder_2(img1_conv1, img1_conv2, img1_conv3, img1_conv4,
                                                                                            warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4)

        return sythsis_output, flow_10_1, flow_10_2, flow_10_3, flow_10_4


class FlowNet_MultiscaleWarp_2Frame_multi(nn.Module):
    ''' 2-frame, using multiple 4 flow '''
    def __init__(self, K):
        super(FlowNet_MultiscaleWarp_2Frame_multi, self).__init__()
        self.K = K
        # self.FlowNetMulti = FlowNetMulti(6, N=self.K)
        self.FlowNetMulti = FlowNetMulti_compact(6, N=self.K)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

        # before computing dot for attention, do a linear transform
        self.conv1_theta = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv1_phi = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv2_theta = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv2_phi = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv3_theta = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv3_phi = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv4_theta = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 
        self.conv4_phi = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0) 

    def attentioned_warp(self, feat_map, flow_list, base_feat_map, theta, phi):
        ''' for every feat_map, use flow_list to compute K warped feature map, 
            then generate attention using base_feat_map,
            finally output attentional weighted warped feature map 
            Args:
                feat_map: [N,C,H,W]
                flow_list: k * [N,2,H,W]
                base_feat_map: [N,C,H,W]
        '''

        base_feat_map_phi = torch.unsqueeze(phi(base_feat_map), 1)  # [N,1,C1,H,W]

        batch_size = base_feat_map_phi.size(0)
        channel_dims = base_feat_map_phi.size(2)

        # for every flow, compute warping, then flatten tensor
        warped_feat_map_theta = [None] * self.K  #  K * [N,C1,H,W]
        warped_feat_map = [None] * self.K  #  K * [N,C,H,W]

        feat_map_theta = theta(feat_map)

        # warped feature map theta K * [N,C1,H,W] ---> (N,K,C1,H,W)
        for i in range(self.K):
            warped_feat_map_theta[i] = torch.unsqueeze(self.Backward_warp(feat_map_theta, flow_list[i]), 1)   # feature K* (N,1,C,H,W) 
        warped_feat_map_theta = torch.cat(warped_feat_map_theta, dim=1)

        # warped feature map K * [N,C,H,W] ---> (N,K,C,H,W)
        for i in range(self.K):
            warped_feat_map[i] = torch.unsqueeze(self.Backward_warp(feat_map, flow_list[i]), 1)   # feature K* (N,1,C,H,W) 
        warped_feat_map = torch.cat(warped_feat_map, dim=1)

        # inner product of warped and original   (N,K,C1,H,W) * (N,1,C1,H,W) /sqrt(C1) = (N,K,C1,H,W) --sum-> (N,K,1,H,W)
        distance = torch.sum((warped_feat_map_theta * base_feat_map_phi)/math.sqrt(channel_dims), 2, keepdim=True)

        # compute attention using softmax   (N,K,1,H,W)
        attention = torch.softmax(distance, dim=1)

        # sum the warped feature map using attention weight    (N,K,1,H,W) * (N,K,C,H,W) = (N,K,C,H,W) --sum-along-dim1-->  (N,C,H,W)
        attentioned_warped = torch.sum(attention * warped_feat_map, 1, keepdim=False)

        return attentioned_warped, attention

    def forward(self, img0, img1, img2):
        ## encode img1
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(img1)

        ## encode img0
        img0_conv1, img0_conv2, img0_conv3, img0_conv4 = self.Encoder(img0)

        # multiple flow 1->0   K * [N,2,H,W]
        flow_list_dict  =  self.FlowNetMulti(img1, img0)
        warp0_conv1, attention1 = self.attentioned_warp(img0_conv1, flow_list_dict['flow_1'], img1_conv1, self.conv1_theta, self.conv1_phi)
        warp0_conv2, _ = self.attentioned_warp(img0_conv2, flow_list_dict['flow_2'], img1_conv2, self.conv2_theta, self.conv2_phi)
        warp0_conv3, _ = self.attentioned_warp(img0_conv3, flow_list_dict['flow_3'], img1_conv3, self.conv3_theta, self.conv3_phi)
        warp0_conv4, _ = self.attentioned_warp(img0_conv4, flow_list_dict['flow_4'], img1_conv4, self.conv4_theta, self.conv4_phi)
        # fusion
        sythsis_output  = self.UNet_decoder_2(img1_conv1, img1_conv2, img1_conv3, img1_conv4,
                                                                                            warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4)

        return tuple([sythsis_output] + flow_list_dict['flow_1'] + [attention1])

class FlowNet_MultiscaleWarp_2Frame_multi_Simple(nn.Module):
    ''' 2-frame, using multiple 4 flow '''
    def __init__(self, K):
        super(FlowNet_MultiscaleWarp_2Frame_multi_Simple, self).__init__()
        self.K = K
        # self.FlowNetMulti = FlowNetMulti(6, N=self.K)
        self.FlowNetMulti = FlowNetMulti_compact(6, N=self.K, compute_attention=True)
        self.Backward_warp = Backward_warp()
        self.Backward_warp1 = Backward_warp_predefined(256, 448)
        self.Backward_warp2 = Backward_warp_predefined(128, 224)
        self.Backward_warp3 = Backward_warp_predefined(64, 112)
        self.Backward_warp4 = Backward_warp_predefined(32, 56)
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def attentioned_warp(self, feat_map, flow_list, flow_softmax_list, warp_fun):
        for i in range(self.K):
            if i==0:
                warped_feat_map = flow_softmax_list[i] * self.Backward_warp(feat_map, flow_list[i])
            else:
                warped_feat_map = warped_feat_map + flow_softmax_list[i] * self.Backward_warp(feat_map, flow_list[i])
        return warped_feat_map

    def forward(self, img0, img1, img2):
        ## encode img1
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(img1)

        ## encode img0
        img0_conv1, img0_conv2, img0_conv3, img0_conv4 = self.Encoder(img0)

        # multiple flow 1->0   K * [N,2,H,W]
        flow_list_dict, flow_softmax_list_dict  =  self.FlowNetMulti(img1, img0)
        # for f in flow_list_dict:
        #     print(f, flow_list_dict[f][0].size())
        # exit()
            
        warp0_conv1 = self.attentioned_warp(img0_conv1, flow_list_dict['flow_1'], flow_softmax_list_dict['flow_1_softmax'],self.Backward_warp1)
        warp0_conv2 = self.attentioned_warp(img0_conv2, flow_list_dict['flow_2'], flow_softmax_list_dict['flow_2_softmax'],self.Backward_warp2)
        warp0_conv3 = self.attentioned_warp(img0_conv3, flow_list_dict['flow_3'], flow_softmax_list_dict['flow_3_softmax'],self.Backward_warp3)
        warp0_conv4 = self.attentioned_warp(img0_conv4, flow_list_dict['flow_4'], flow_softmax_list_dict['flow_4_softmax'],self.Backward_warp4)
        # fusion
        sythsis_output  = self.UNet_decoder_2(img1_conv1, img1_conv2, img1_conv3, img1_conv4,
                                                                                            warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4)

        return tuple([sythsis_output] + flow_list_dict['flow_1'] + flow_softmax_list_dict['flow_1_softmax'])


class FlowNet_MultiscaleWarp_2Frame_multi_Simple_efficient(nn.Module):
    ''' 2-frame, using multiple 4 flow '''
    def __init__(self, K):
        super(FlowNet_MultiscaleWarp_2Frame_multi_Simple_efficient, self).__init__()
        self.K = K
        # self.FlowNetMulti = FlowNetMulti(6, N=self.K)
        self.FlowNetMulti = FlowNetMulti_compact(6, N=self.K, compute_attention=True, return_list=False)
        self.Backward_warp_multi = Backward_warp_multi()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def attentioned_warp(self, feat_map, flow_list, flow_softmax_list):
        for i in range(self.K):
            if i==0:
                warped_feat_map = flow_softmax_list[i] * self.Backward_warp(feat_map, flow_list[i])
            else:
                warped_feat_map = warped_feat_map + flow_softmax_list[i] * self.Backward_warp(feat_map, flow_list[i])
        return warped_feat_map

    def forward(self, img0, img1, img2):
        ## encode img1
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(img1)

        ## encode img0
        img0_conv1, img0_conv2, img0_conv3, img0_conv4 = self.Encoder(img0)

        # multiple flow 1->0   K * [N,2,H,W]
        flow_list_dict, flow_softmax_list_dict  =  self.FlowNetMulti(img1, img0)
        # print(img0_conv1.size(), flow_list_dict['flow_1'].size(), flow_softmax_list_dict['flow_1_softmax'].size())
        warp0_conv1 = self.Backward_warp_multi(img0_conv1, flow_list_dict['flow_1'], flow_softmax_list_dict['flow_1_softmax'])
        warp0_conv2 = self.Backward_warp_multi(img0_conv2, flow_list_dict['flow_2'], flow_softmax_list_dict['flow_2_softmax'])
        warp0_conv3 = self.Backward_warp_multi(img0_conv3, flow_list_dict['flow_3'], flow_softmax_list_dict['flow_3_softmax'])
        warp0_conv4 = self.Backward_warp_multi(img0_conv4, flow_list_dict['flow_4'], flow_softmax_list_dict['flow_4_softmax'])
        # fusion
        sythsis_output  = self.UNet_decoder_2(img1_conv1, img1_conv2, img1_conv3, img1_conv4, 
                                                warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4)

        return sythsis_output, flow_list_dict['flow_1'], flow_softmax_list_dict['flow_1_softmax']



class FlowNet_MultiscaleWarp_2Frame_grid(nn.Module):
    ''' 2-frame, using multiple 4 flow '''
    def __init__(self, K):
        super(FlowNet_MultiscaleWarp_2Frame_multi, self).__init__()
        self.K = K
        self.FlowNet = FlowNet(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def attentioned_warp(self, feat_map, flow_list, base_feat_map):
        ''' for every feat_map, use flow_list to compute K warped feature map, 
            then generate attention using base_feat_map,
            finally output attentional weighted warped feature map '''


        batch_size = feat_map.size(0)
        channel_dims = feat_map.size(1)

        # for every flow, compute warping, then flatten tensor
        warped_feat_map = [None] * self.K  #  K * [N,C,H,W]

        # warped feature map  K * [N,C,H,W] ---> (N,K,H,W,C)
        for i in range(self.K):
            warped_feat_map[i] = torch.unsqueeze(self.Backward_warp(feat_map, flow_list[i]), 1)   # feature K* (N,1,C,H,W) 
        warped_feat_map = torch.cat(warped_feat_map, dim=1)

        # inner product of warped and original   (N,K,H,W,C) * (N,1,H,W,C) /sqrt(K) = (N,K,H,W,C) --sum-> (N,K,H,W,1)
        distance = torch.sum((warped_feat_map * torch.unsqueeze(base_feat_map, 1))/math.sqrt(channel_dims), 4, keepdim=True)

        # compute attention using softmax   (N,K,H,W,1)
        attention = torch.softmax(distance, dim=1)

        # sum the warped feature map using attention weight    (N,K,H,W,1) * (N,K,H,W,C) = (N,K,H,W,C) --sum-along-dim1-->  (N,H,W,C)
        attentioned_warped = torch.sum(attention * warped_feat_map, 1, keepdim=False)

        return attentioned_warped

    def forward(self, img0, img1, img2):
        ## encode img1
        img1_conv1, img1_conv2, img1_conv3, img1_conv4 = self.Encoder(img1)

        ## encode img0
        img0_conv1, img0_conv2, img0_conv3, img0_conv4 = self.Encoder(img0)

        ## 
        flow_10 = self.FlowNet(img1, img0)
        flow_10_1, flow_10_2, flow_10_3, flow_10_4 = flow_10['flow_1'],  flow_10['flow_2'],  flow_10['flow_3'], flow_10['flow_4']

        # multiple flow    K * [N,2,H,W]
        flow_list_dict  =  self.FlowNetMulti(img1, img0)
        warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4 =  self.attentioned_warp(img0_conv1, flow_list_dict['flow_1'], img1_conv1),     \
                                                                self.attentioned_warp(img0_conv2, flow_list_dict['flow_2'], img1_conv2),    \
                                                                self.attentioned_warp(img0_conv3, flow_list_dict['flow_3'], img1_conv3),    \
                                                                self.attentioned_warp(img0_conv4, flow_list_dict['flow_4'], img1_conv4)
        # fusion
        sythsis_output  = self.UNet_decoder_2(img1_conv1, img1_conv2, img1_conv3, img1_conv4,
                                                                                            warp0_conv1, warp0_conv2, warp0_conv3, warp0_conv4)

        return tuple([sythsis_output] + flow_list_dict['flow_1'])