import torch
import numpy as np
from opt import config_parser
import os
from tqdm.auto import tqdm
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataLoader import dataset_dict
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_VM_model(args):
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return
    ckpt = torch.load(args.ckpt,map_location = device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)
    return tensorf
def distill(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray


    # init log file
    logfolder = f'{args.basedir}/{args.expname}'
    os.makedirs(f'{logfolder}/distill', exist_ok=True)
    #os.makedirs(f'{logfolder}/img_vis',exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # load_VM_ckpt
    model_VM = load_VM_model(args)
    print(model_VM)

    # init resolution?
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    # init CP paramaeters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))#sqrt(128**2+128**2+128**2)/step(0.5)=443, nSample never more than 443 samples for one pixel
    tensorf = eval(args.distill_model_name)(aabb, reso_cur, device,
                                    density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh,
                                    app_dim=args.data_dim_color, near_far=near_far,
                                    shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre,
                                    density_shift=args.density_shift, distance_scale=args.distance_scale,
                                    pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe,
                                    featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
    args = config_parser()
    print(args)
    distill(args)