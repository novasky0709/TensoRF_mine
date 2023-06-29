import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from student_model.base_nerf import raw2alpha, BaseNeRF


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim
        torch.nn.Linear
        for i in range(self.z_num):
            self.z_list.append(nn.Parameter(torch.randn(self.z_dim)))


class HyperMLP(nn.Module):

    def __init__(self, z_dim = 32, in_size=256, out_size=256, K = 7,scene_num = 2):
        super(HyperMLP, self).__init__()
        # a HyperMLP is a mlp which param size is out_size * (in_size + 1)
        # which is split into B * C * W (which are two matrix)
        # B.size [in_size,K] (generate from z_dim) parameter size: z_dim * (in_size * K + 1)
        # C.size [K,K] (per task one matrix C) parameter size: K * K
        # W.size [K, in_size + 1] (share across tasks) parameter size: K * (in_size + 1)
        # for multiple task ,the para size of vanilla MLP is : N * out_size * (in_size + 1)
        # for multiple task ,the para size of Hyper MLP is : N * (K * K + z_dim * (in_size * K + 1)) + K * (in_size + 1)
        # if B is share across task, the para size of Hyper MLP is : N * (K * K ) + (K * (in_size + 1) +  z_dim * (in_size * K + 1))
        # e.g. K = 5 z = 16 insize=outsize=256 vanilla: N*65792 Hyper:N*20521 + 1285
        # e.g. K = 32 z = 4 insize=outsize=256 vanilla: N*65792 Hyper:N*33796 + 8224
        # e.g. [B is share weight]  K = 5 z = 16 insize=outsize=256, Hyper: N*65792 Hyper:N*25 + 21781
        # e.g. [B is share weight] K = 16 z = 64 insize=outsize=256, Hyper: N*65792 Hyper:N*256 + 266320
        # e.g. [B is share weight] K = 32 z = 16 insize=outsize=256, Hyper: N*65792 Hyper:N*1024 + 139312 (这个差不多了,参数量,初始两倍nerf，然后基本没有)
        self.z_dim = z_dim
        self.out_size = out_size
        self.in_size = in_size
        self.K = K
        self.C_w_list = nn.ParameterList()
        self.C_b_list = nn.ParameterList()
        self.B_w = nn.Parameter(torch.rand(self.z_dim, (self.in_size) * self.K))
        torch.nn.init.normal_(self.B_w, 0.0, np.sqrt(2) / np.sqrt((self.in_size) * self.K))

        self.B_b = nn.Parameter(torch.rand((self.in_size) * self.K))
        torch.nn.init.constant_(self.B_b, 0.0)

        for scene_id in range(scene_num):

            C_w = nn.Parameter(torch.rand(self.K , self.K))
            torch.nn.init.normal_(C_w, 0.0, np.sqrt(2) / np.sqrt(self.K))
            self.C_w_list.append(C_w)

            C_b = nn.Parameter(torch.rand(self.K))
            torch.nn.init.constant_(C_b, 0.0)
            self.C_b_list.append(C_b)


        self.W_w = nn.Parameter(torch.rand(self.K , self.out_size))
        torch.nn.init.normal_(self.W_w, 0.0, np.sqrt(2) / np.sqrt(self.out_size))
        self.W_b = nn.Parameter(torch.rand(self.out_size))
        torch.nn.init.constant_(self.W_b, 0.0)
    def forward(self, z,scene_id = 0):

        B = torch.matmul(z, self.B_w) + self.B_b
        # B = F.tanh(B)
        B = B.view(self.in_size, self.K) # [in_size,K]

        # mlp_para = F.tanh(torch.matmul(torch.relu(torch.matmul(B, self.C)),self.W))
        mlp_para = torch.matmul((torch.matmul(B, self.C_w_list[scene_id])+self.C_b_list[scene_id]), self.W_w)
        # mlp_para = mlp_para.view(self.in_size, self.out_size + 1)

        return mlp_para, self.W_b


class CrossSceneHypernetVanillaNeRFBeta(BaseNeRF):
    def __init__(self, aabb, gridSize,scene_num, D = 8,W = 256, device = 'cuda:0',pos_pe = 10, dir_pe = 4,distance_scale =25, \
                 rayMarch_weight_thres = 0.0001, near_far=[2.0, 6.0], density_shift = -10,step_ratio = 0.5,z_num = 1, z_dim = 16, c_dim = 11):
        super(CrossSceneHypernetVanillaNeRFBeta,self).__init__(aabb, gridSize, density_shift, near_far, device, step_ratio, rayMarch_weight_thres, distance_scale)
        self.scene_num = scene_num
        self.D = D
        self.W = W
        self.c_dim = c_dim
        self.device = device
        self.pos_pe = pos_pe
        self.dir_pe = dir_pe
        self.z_num = z_num
        self.z_dim = z_dim
        self.pos_embed_fn, pos_dim_pe = utils.get_embedder(self.pos_pe, 0, input_dims=3)
        self.dir_embed_fn, dir_dim_pe = utils.get_embedder(self.dir_pe, 0, input_dims=3)

        self.init_nn(pos_dim_pe, dir_dim_pe)
        self.set_device(device)
    def forward(self,ray_sampled, xyz_sampled, viewdir_sampled, z_vals, ray_valid,white_bg=True, is_train=False, ndc_ray=False,scene_id=0):
        input_pos = self.pos_embed_fn(xyz_sampled[ray_valid])

        dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)
        app_feat = torch.zeros((*xyz_sampled.shape[:2], 256), device=xyz_sampled.device)
        sigma_feature = None

        if ray_valid.any():
            x = input_pos
            for i,l in enumerate(self.encoder):
                w, b = self.encoder[i](self.z_space_list[scene_id].z_list[i],scene_id)
                x = torch.matmul(x,w) + b
                x = F.relu(x)
                if i in [self.D // 2]:
                    x = torch.cat([input_pos,x],dim = -1)
            hidden_feat = x
            # validsigma = F.softplus(self.density_linear(x) + self.density_shift)
            validsigma = F.relu(self.density_linear_list[scene_id](x))
            sigma[ray_valid] = validsigma.squeeze(dim=-1)
            app_feat[ray_valid] = hidden_feat
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)  # distance_scale 25
        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            input_dir = self.dir_embed_fn(viewdir_sampled[app_mask])
            x = self.app_linear_list[scene_id][0](app_feat[app_mask])
            app_feat_valid = torch.cat([input_dir, x], dim=-1)
            x = F.relu(self.app_linear_list[scene_id][1](app_feat_valid))
            valid_rgbs = torch.sigmoid(self.app_linear_list[scene_id][2](x))
            rgb[app_mask] = valid_rgbs
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)
        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0, 1)
        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * ray_sampled[..., -1]

        return rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight, sigma_feature, app_feat
        # return sigma, app, hidden_feat

    def init_nn(self,pos_dim_pe, dir_dim_pe):
       self.z_space_list = torch.nn.ModuleList()
       self.density_linear_list = torch.nn.ModuleList()
       self.app_linear_list = torch.nn.ModuleList()
       self.encoder = nn.ModuleList([HyperMLP(self.z_dim, pos_dim_pe, self.W, self.c_dim, self.scene_num)] + \
                               [HyperMLP(self.z_dim, self.W, self.W, self.c_dim, self.scene_num) if (
                                           i not in [self.D // 2]) else
                                HyperMLP(self.z_dim, self.W + pos_dim_pe, self.W, self.c_dim, self.scene_num) for i in
                                range(self.D - 1)]
                               )

       for scene_id in range(self.scene_num):
           self.z_space_list.append(Embedding(self.D , self.z_dim))

           density_linear = nn.Linear(self.W, 1)
           torch.nn.init.constant_(density_linear.bias, 0.0)
           torch.nn.init.normal_(density_linear.weight, 0.0, np.sqrt(2) / np.sqrt(1))
           self.density_linear_list.append(density_linear)
           app_linear = nn.ModuleList(
                [nn.Linear(self.W, self.W)] + [nn.Linear(self.W + dir_dim_pe, self.W // 2)] + [nn.Linear(self.W // 2, 3)])
           torch.nn.init.constant_(app_linear[0].bias, 0.0)
           torch.nn.init.normal_(app_linear[0].weight, 0.0, np.sqrt(2) / np.sqrt(self.W))
           torch.nn.init.constant_(app_linear[1].bias, 0.0)
           torch.nn.init.normal_(app_linear[1].weight, 0.0, np.sqrt(2) / np.sqrt(self.W//2))
           torch.nn.init.constant_(app_linear[2].bias, 0.0)
           torch.nn.init.normal_(app_linear[2].weight, 0.0, np.sqrt(2) / np.sqrt(3))
           self.app_linear_list.append(app_linear)
    def get_optparam_groups(self, lr_init_network = 0.001,lr_init_paras = 0.001):
        # {'params': self.z_space.z_list, 'lr': lr_init_network},
        grad_vars = [ {'params': self.encoder.parameters(), 'lr': lr_init_paras},
                         {'params': self.density_linear_list.parameters(), 'lr':lr_init_network},{'params': self.z_space_list.parameters(), 'lr':lr_init_paras},{'params': self.app_linear_list.parameters(), 'lr':lr_init_network}]
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
            'z_num':self.z_num,
            'z_dim':self.z_dim,
            'c_dim':self.c_dim,
            'scene_num':self.scene_num
        }

    def load(self, ckpt):
        # we need re-implementation this function. e.g.
        # self.load_state_dict(ckpt['state_dict'],strict=False)
        missing, _ = self.load_state_dict(ckpt['state_dict'],strict=False)
if __name__ == '__main__':
    # emb = Embedding(1,32)
    # hmlp = HyperMLP(z_dim = 32, in_size=256, out_size=64, K = 7)
    # w,b = hmlp(emb.z_list[0])
    # print(w.shape,b.shape)

    from opt import config_parser
    from models.tensoRF import TensorVMSplit_Distill
    args = config_parser()
    device = 'cuda:0'
    len_fitted_scene = 0
    for len_fitted_scene in range(3):
        args.ckpt = '/home/yuze/Documents/project/TensoRF/log/0TensoRF_model/lego.th'
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        tensorf_tea = TensorVMSplit_Distill(**kwargs)
        stu_args = {'scene_num':len_fitted_scene + 1,'distance_scale': tensorf_tea.distance_scale,
                    'rayMarch_weight_thres': ckpt['kwargs']['rayMarch_weight_thres'],
                    'aabb': tensorf_tea.aabb, 'gridSize': tensorf_tea.gridSize, 'near_far': tensorf_tea.near_far
            , 'density_shift': tensorf_tea.density_shift, 'step_ratio': tensorf_tea.step_ratio, 'device': device
            , 'pos_pe': args.dis_network_pos_pe, 'dir_pe': args.dis_network_dir_pe, 'z_dim': args.dis_network_z_dim,
                    'c_dim': args.dis_network_c_dim, }
        stu_model = CrossSceneHypernetVanillaNeRFBeta(**stu_args)
        # ckpt_student = torch.load('/home/yuze/Downloads/2_scene.th', map_location=device)
        # stu_model.load(ckpt_student)
        stu_model.save('/home/yuze/Downloads/{}_scenes.th'.format(len_fitted_scene))
    # rays_chunk = torch.rand([128,6] ,device = 'cuda:0')
    # xyz_chunk = torch.rand([128,1036,3] ,device = 'cuda:0')
    # viewdir_chunk = torch.rand([128,1036,3] ,device = 'cuda:0')
    # z_vals_chunk = torch.rand([128,1036] ,device = 'cuda:0')
    # ray_valid_chunk = z_vals_chunk>0
    # for id in range(len_fitted_scene + 1):
    #     rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight, sigma_feat, app_feat = stu_model(rays_chunk, xyz_chunk, viewdir_chunk, z_vals_chunk, ray_valid_chunk,is_train=True, white_bg = True, ndc_ray=0,scene_id = id)
    #     print(rgb_map.shape)
    #     print(rgb.shape)
    #     print(sigma.shape)
    # alpha_gt = torch.ones_like(alpha)
    # loss = torch.mean((alpha_gt - alpha)**2)
    # #loss = torch.log(sigma.sum())
    # loss.backward() # 63

    # output = hvn.get_optparam_groups()
    # print(output)