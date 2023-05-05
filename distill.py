# distill TensoRF to another nerf arch, such as vanilla MLP

import os

import torch
from tqdm.auto import tqdm
from opt import config_parser

import json, random
from renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from student_model.vanilla_nerf import VanillaNeRF
from dataLoader import dataset_dict
import sys
from models.tensoRF import TensorVMSplit_Distill
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tea_renderer = Distill_renderer
stu_renderer = Stu_vanilla_renderer

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr += self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr + self.batch]



def distill(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    # upsamp_list = args.upsamp_list
    # update_AlphaMask_list = args.update_AlphaMask_list if args.update_AlphaMask_list is not None else []
    # n_lamb_sigma = args.n_lamb_sigma
    # n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/distill/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/distill/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/distill/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,
                                                args.step_ratio))  # sqrt(128**2+128**2+128**2)/step(0.5)=443, nSample never more than 443 samples for one pixel

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!! Distill must from a ckpt')
        return
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf_tea = eval(args.model_name+"_Distill")(**kwargs)
    tensorf_tea.load(ckpt)
    stu_args = {'distance_scale':args.distance_scale,'rayMarch_weight_thres':args.rm_weight_mask_thre}


    stu_model = eval(args.stu_model_name)(**stu_args)

    grad_vars = stu_model.get_optparam_groups(args.dis_lr_init)
    if args.dis_lr_decay_iters > 0:
        lr_factor = args.dis_lr_decay_target_ratio ** (1 / args.dis_lr_decay_iters)
    else:
        args.dis_lr_decay_iters = args.dis_n_iters
        lr_factor = args.dis_lr_decay_target_ratio ** (1 / args.dis_n_iters)
    loss_hyperparam = {'dis_start_appfeatloss_iter':args.dis_start_appfeatloss_iter,
                       'dis_end_appfeatloss_iter': args.dis_end_appfeatloss_iter,
                        'dis_start_rfloss_iter':args.dis_start_rfloss_iter,
                       'dis_end_rfloss_iter':args.dis_end_rfloss_iter,
                       'dis_start_ftloss_iter':args.dis_start_ftloss_iter,
                       'dis_end_ftloss_iter':args.dis_end_ftloss_iter,
                       'dis_appfeatloss_weight':args.dis_appfeatloss_weight,
                       'dis_rfloss_weight':args.dis_rfloss_weight,
                       'dis_ftloss_weight': args.dis_ftloss_weight
                       }
    for k in loss_hyperparam.keys():
        if  k.endswith('_iter') and loss_hyperparam[k] == -1 :
            loss_hyperparam[k] = args.dis_n_iters + 1

    print("lr decay", args.dis_lr_decay_target_ratio, args.dis_lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    # N_voxel_list = (torch.round(torch.exp(
    #     torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[
    #                1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf_tea.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    # init loss
    # Ortho_reg_weight = args.Ortho_weight
    # print("initial Ortho_reg_weight", Ortho_reg_weight)
    #
    # L1_reg_weight = args.L1_weight_inital
    # print("initial L1_reg_weight", L1_reg_weight)
    # TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    # tvreg = TVLoss()
    # print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")

    pbar = tqdm(range(args.dis_n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        # data example:
        # rays_train [4096,6], rgb_train [4096,3], rgb_maps [4096,3] depth_maps [4096]
        with torch.no_grad():
            tea_rgb_maps, tea_depth_maps, tea_rgbs, tea_sigmas, tea_alphas, tea_density_feats, tea_app_feats, xyz_sampled,viewdirs , z_vals, ray_valid  = tea_renderer(rays_train, tensorf_tea, chunk=args.batch_size,
                                                                            N_samples=nSamples, white_bg=white_bg,
                                                                            ndc_ray=ndc_ray, device=device, is_train=False)
            #e.g. rgb_maps [4096,3];depth_maps [4096]; rgbs [4096,443,3]; sigmas,alphas: [4096,443]
            # app_feats [19817,27]->youwenti xyz_sampled [4096,443,3]; z_vals [4096,443]; ray_valid [4096,443]
        stu_rgb_maps, stu_depth_maps, stu_rgbs, stu_sigmas, stu_alphas, _, stu_app_feats = stu_renderer(stu_model, rays_train, xyz_sampled, viewdirs, z_vals, ray_valid,  chunk=args.dis_batch_size,  ndc_ray=ndc_ray, white_bg = white_bg, is_train=True, device = device)

        # loss
        total_loss = 0
        # cauculate loss
        if (iteration + 1 > loss_hyperparam['dis_start_appfeatloss_iter']) and  (iteration + 1 < loss_hyperparam['dis_end_appfeatloss_iter']):
            assert(tea_app_feats.shape == stu_app_feats.shape),'app_feat size dont match between student and teacher'
            total_loss += loss_hyperparam['dis_appfeatloss_weight'] * torch.mean((tea_app_feats - stu_app_feats)**2)
        if (iteration + 1 > loss_hyperparam['dis_start_rfloss_iter']) and  (iteration + 1 < loss_hyperparam['dis_end_rfloss_iter']):
            assert (tea_sigmas.shape == stu_sigmas.shape) and (tea_rgbs.shape == stu_rgbs.shape), 'app_feat size dont match between student and teacher'
            total_loss += loss_hyperparam['dis_rfloss_weight'] * (torch.mean((tea_sigmas - stu_sigmas) ** 2) + torch.mean((tea_rgbs - stu_rgbs) ** 2))
        if (iteration + 1 > loss_hyperparam['dis_start_ftloss_iter']) and  (iteration + 1 < loss_hyperparam['dis_end_ftloss_iter']):
            assert (tea_rgb_maps.shape == stu_rgb_maps.shape), 'app_feat size dont match between student and teacher'
            total_loss += loss_hyperparam['dis_ftloss_weight'] * torch.mean((stu_rgb_maps - tea_rgb_maps) ** 2)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        mse = torch.mean((stu_rgb_maps - tea_rgb_maps) ** 2).detach().item()

        PSNRs.append(-10.0 * np.log(mse) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', mse, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {mse:.6f}'
            )
            PSNRs = []

        # if iteration % args.vis_every == args.vis_every - 1 and args.N_vis != 0:
        #     PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
        #                             prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
        #                             compute_extra_metrics=False)
        #     summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        # if iteration in update_AlphaMask_list:
        #     reso_mask = reso_cur
        #     # if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
        #     #     reso_mask = reso_cur
        #     new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
        #     if iteration == update_AlphaMask_list[0]:
        #         tensorf.shrink(new_aabb)
        #         # tensorVM.alphaMask = None
        #         L1_reg_weight = args.L1_weight_rest
        #         print("continuing L1_reg_weight", L1_reg_weight)
        #
        #     if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
        #         # filter rays outside the bbox
        #         allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs)
        #         trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)
        #
        # if iteration in upsamp_list:
        #     n_voxels = N_voxel_list.pop(0)
        #     reso_cur = N_to_reso(n_voxels, tensorf.aabb)
        #     nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
        #     tensorf.upsample_volume_grid(reso_cur)

            # if args.lr_upsample_reset:
            #     print("reset lr to initial")
            #     lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            # else:
            #     lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            # grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
            # optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    stu_model.save(f'{logfolder}/distill_{args.expname}.th')

    # if args.render_train:
    #     os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
    #     train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
    #     PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
    #                             N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    #     print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
    #
    # if args.render_test:
    #     os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
    #     PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
    #                             N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
    #     summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
    #     print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
    #
    # if args.render_path:
    #     c2ws = test_dataset.render_path
    #     # c2ws = test_dataset.poses
    #     print('========>', c2ws.shape)
    #     os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
    #     evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
    #                     N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    print(args)

    distill(args)

