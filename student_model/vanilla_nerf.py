import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from student_model.base_nerf import raw2alpha, BaseNeRF

class VanillaNeRF(BaseNeRF):
    def __init__(self, aabb, gridSize, D = 8,W = 256, device = 'cuda:0',pos_pe = 10, dir_pe = 4,distance_scale =25, rayMarch_weight_thres = 0.0001, near_far=[2.0, 6.0], density_shift = -10,step_ratio = 0.5):
        super(VanillaNeRF,self).__init__(aabb, gridSize, density_shift, near_far, device, step_ratio, rayMarch_weight_thres, distance_scale)
        self.D = D
        self.W = W
        self.device = device
        self.pos_pe = pos_pe
        self.dir_pe = dir_pe
        self.pos_embed_fn, pos_dim_pe = utils.get_embedder(self.pos_pe, 0, input_dims=3)
        self.dir_embed_fn, dir_dim_pe = utils.get_embedder(self.dir_pe, 0, input_dims=3)

        self.init_nn(pos_dim_pe, dir_dim_pe)
        self.set_device(device)
    def forward(self,ray_sampled, xyz_sampled, viewdir_sampled, z_vals, ray_valid,white_bg=True, is_train=False, ndc_ray=False):
        input_pos = self.pos_embed_fn(xyz_sampled[ray_valid])

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        app_feat = torch.zeros((*xyz_sampled.shape[:2], 256), device=xyz_sampled.device)
        sigma_feature = None

        if ray_valid.any():
            x = input_pos
            for i,l in enumerate(self.encoder):
                x = self.encoder[i](x)
                x = F.relu(x)
                if i in [self.D // 2]:
                    x = torch.cat([input_pos,x],dim = -1)
            hidden_feat = x
            # validsigma = F.softplus(self.density_linear(x) + self.density_shift)
            validsigma = F.relu(self.density_linear(x))
            sigma[ray_valid] = validsigma.squeeze(dim=-1)
            app_feat[ray_valid] = hidden_feat
        alpha, weight, bg_weight = raw2alpha(sigma, dists*self.distance_scale)# distance_scale 25
        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            input_dir = self.dir_embed_fn(viewdir_sampled[app_mask])
            x = self.app_linear[0](app_feat[app_mask])
            app_feat_valid = torch.cat([input_dir, x],dim = -1)
            x = F.relu(self.app_linear[1](app_feat_valid))
            valid_rgbs =  torch.sigmoid(self.app_linear[2](x))
            rgb[app_mask] = valid_rgbs
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * ray_sampled[..., -1]

        return rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight, sigma_feature, app_feat
        # return sigma, app, hidden_feat


    def init_nn(self,pos_dim_pe, dir_dim_pe):
       self.encoder = nn.ModuleList([nn.Linear(pos_dim_pe,self.W)] + [nn.Linear(self.W,self.W) if (i not in [self.D//2 ]) else nn.Linear(self.W + pos_dim_pe,self.W) for i in range(self.D - 1)])# + [nn.Linear(self.W , self.W)])
       self.density_linear = nn.Linear(self.W,1)
       self.app_linear = nn.ModuleList([nn.Linear(self.W, self.W)] + [nn.Linear(self.W + dir_dim_pe ,self.W //2)] + [nn.Linear(self.W//2,3)])

    def get_optparam_groups(self, lr_init_network = 0.001):
        grad_vars = [{'params': self.encoder.parameters(), 'lr': lr_init_network}, {'params': self.density_linear.parameters(), 'lr': lr_init_network},
                         {'params': self.app_linear.parameters(), 'lr':lr_init_network}]
        return grad_vars
    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'D': self.D,
            'W': self.W,
            'pos_pe': self.pos_pe,
            'dir_pe':self.dir_pe,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'near_far': self.near_far,
            'density_shift': self.density_shift,
            'step_ratio': self.step_ratio,
        }

if __name__ == '__main__':
    # rays_train [4096,6], rgb_train [4096,3], rgb_maps [4096,3] depth_maps [4096]
    from opt import config_parser
    from models.tensoRF import TensorVMSplit_Distill
    args = config_parser()
    device = 'cuda:0'
    args.ckpt = '/home/yuze/Documents/project/TensoRF/log/tensorf_lego_VM_nomsk.th'
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf_tea = TensorVMSplit_Distill(**kwargs)
    tensorf_tea.load(ckpt)
    stu_args = {'distance_scale': tensorf_tea.distance_scale,
                'rayMarch_weight_thres': ckpt['kwargs']['rayMarch_weight_thres'], \
                'aabb': tensorf_tea.aabb, 'gridSize': tensorf_tea.gridSize, 'near_far': tensorf_tea.near_far
        , 'density_shift': tensorf_tea.density_shift, 'step_ratio': tensorf_tea.step_ratio}
    vn = VanillaNeRF(**stu_args)
    rays_chunk = torch.rand([128,6] ,device = 'cuda:0')
    xyz_chunk = torch.rand([128,1036,3] ,device = 'cuda:0')
    viewdir_chunk = torch.rand([128,1036,3] ,device = 'cuda:0')
    z_vals_chunk = torch.rand([128,1036] ,device = 'cuda:0')
    ray_valid_chunk = z_vals_chunk>0
    rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight, sigma_feat, app_feat = vn(rays_chunk, xyz_chunk, viewdir_chunk, z_vals_chunk, ray_valid_chunk,is_train=True, white_bg = True, ndc_ray=0)
    print(rgb_map.shape)
    print(rgb.shape)
    print(sigma.shape)
