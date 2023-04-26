import torch.nn

import utils
from morph_models.residue_models.residueBase import ResidueBase
from utils import  *

class ResidueTensorVMSplit(ResidueBase):
    def __init__(self, src_kwargs, dst_kwargs,device = 'cuda:0',density_D = 8, density_W = 256, app_D = 8, app_W = 256, density_pe = 5, app_pe = 3):
        super(ResidueTensorVMSplit, self).__init__(src_kwargs,dst_kwargs)
        self.device = device
        self.density_D = density_D
        self.density_W = density_W
        self.app_D = app_D
        self.app_W = app_W
        self.density_pe = density_pe
        self.app_pe = app_pe
        self.src_density_dim = 1
        self.src_density_app = 3# src_kwargs['app_dim'] # [27]
        self.density_embed_fn, src_density_dim_pe = utils.get_embedder(density_pe, 0, input_dims=1)
        self.app_embed_fn, src_app_dim_pe = utils.get_embedder(app_pe, 0, input_dims=3) #src_kwargs['app_dim'])
        self.dst_density_dim = 1
        self.dst_app_dim = 3# dst_kwargs['app_dim']  # 27
        self.density_mlp = self.init_nn(src_density_dim_pe,self.dst_density_dim,density_D,density_W,density_pe)
        self.app_mlp = self.init_nn(src_app_dim_pe,self.dst_app_dim,app_D,app_W,app_pe)

    def forward(self,x):
        input_densitys, input_apps = torch.split(x,[self.src_density_dim, self.src_density_app],dim = -1)
        input_densitys = self.density_embed_fn(input_densitys)
        input_apps = self.app_embed_fn(input_apps)
        x = input_densitys
        for i,l in enumerate(self.density_mlp):
            if i in [self.density_D//2]:
                x = torch.cat([input_densitys,x], dim = -1)
            x = self.density_mlp[i](x)
            x = F.relu(x)
        output_densitys = x
        x = input_apps
        for i,l in enumerate(self.app_mlp):
            if i in [self.app_D//2]:
                x = torch.cat([input_apps,x], dim = -1)
            x = self.app_mlp[i](x)
            x = F.relu(x)
        output_apps = x
        return torch.cat([output_densitys, output_apps],dim = -1)


    def init_nn(self,input_dim, output_dim, D, W, pe):
        #torch.nn.Linear()
        return nn.ModuleList([nn.Linear(input_dim,W//2)] + [nn.Linear(W//2,W)] + [nn.Linear(W,W) if (i not in [D//2 - 2]) else nn.Linear(W + input_dim,W) for i in range(D-3)]  + [nn.Linear(W,output_dim)]).to(self.device)


    def get_optparam_groups(self, lr_init_network=0.001):
        grad_vars = [{'params': self.density_mlp.parameters(), 'lr': lr_init_network}, {'params': self.app_mlp.parameters(), 'lr': lr_init_network}]
        return grad_vars

if __name__ == '__main__':
    src_kwargs = {'app_dim':3}
    dst_kwargs = {'app_dim':3}
    model = ResidueTensorVMSplit(src_kwargs,dst_kwargs)
    input  = torch.rand([12000,4]).to('cuda:0')# [b, n, den(1) + app(27)]
    output = model(input)
    d,c = torch.split(output,[1,3],dim = -1)
    print(d.squeeze(-1).shape,c.shape)
