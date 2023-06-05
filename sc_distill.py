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
from student_model.hypernet_vanilla_nerf import HypernetVanillaNeRF
from student_model.cs_hypernet_vanilla_nerf import CrossSceneHypernetVanillaNeRF
from dataLoader import dataset_dict
import sys
from models.tensoRF import TensorVMSplit_Distill
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tea_renderer = Distill_renderer
stu_renderer = Stu_vanilla_renderer
stu_renderer_test = Stu_vanilla_renderer_test
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


@torch.no_grad()
def evaluation_student_model(test_dataset,stu_model, args, stu_renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):

    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass
    for scene_id in range(len(test_dataset)):
        PSNRs, rgb_maps, depth_maps = [], [], []
        ssims, l_alex, l_vgg = [], [], []
        near_far = test_dataset[scene_id].near_far
        img_eval_interval = 1 if N_vis < 0 else max(test_dataset[scene_id].all_rays.shape[0] // N_vis,1)
        idxs = list(range(0, test_dataset[scene_id].all_rays.shape[0], img_eval_interval))
        for idx, samples in tqdm(enumerate(test_dataset[scene_id].all_rays[0::img_eval_interval]), file=sys.stdout):
            W, H = test_dataset[scene_id].img_wh
            rays = samples.view(-1,samples.shape[-1])
            rgb_map, _, depth_map, _, _ = stu_renderer_test(rays, stu_model, chunk=args.dis_batch_size, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device,scene_id = scene_id)
            rgb_map = rgb_map.clamp(0.0, 1.0)

            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

            depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
            if len(test_dataset[scene_id].all_rgbs):
                gt_rgb = test_dataset[scene_id].all_rgbs[idxs[idx]].view(H, W, 3)
                loss = torch.mean((rgb_map - gt_rgb) ** 2)
                PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))
            rgb_map = (rgb_map.numpy() * 255).astype('uint8')
            # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            rgb_maps.append(rgb_map)
            depth_maps.append(depth_map)
            if savePath is not None:
                imageio.imwrite(f'{savePath}/{prtx}_scene{scene_id}_{idx:03d}.png', rgb_map)
                rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
                imageio.imwrite(f'{savePath}/rgbd/_scene{scene_id}_{prtx}{idx:03d}.png', rgb_map)

        # imageio.mimwrite(f'{savePath}/{prtx}_scene{scene_id}_video.mp4', np.stack(rgb_maps), fps=30, quality=10)
        # imageio.mimwrite(f'{savePath}/{prtx}_scene{scene_id}_depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

        if PSNRs:
            psnr = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                np.savetxt(f'{savePath}/{prtx}_scene{scene_id}_mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
            else:
                np.savetxt(f'{savePath}/{prtx}_scene{scene_id}_mean.txt', np.asarray([psnr]))
    return PSNRs

def distill(args):
    # init dataset
    print(args.sc_datadir_list)
    if args.sc_datadir_list is None:
        len_fitted_scene = 0
    else:
        len_fitted_scene = len(args.sc_datadir_list)
    print('{} scenes have been fit.'.format(len_fitted_scene))
    train_dataset = []
    test_dataset = []
    if len_fitted_scene !=0:
        for datadir in args.sc_datadir_list:
            print('Loading {}'.format(datadir))
            dataset = dataset_dict[args.dataset_name]
            train_dataset.append(dataset(datadir, split='train', downsample=args.downsample_train, is_stack=False))
            test_dataset.append(dataset(datadir, split='test', downsample=args.downsample_train, is_stack=True))
    dataset = dataset_dict[args.dataset_name]
    train_dataset.append(dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False))
    test_dataset.append(dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True))
    white_bg = train_dataset[len_fitted_scene].white_bg
    near_far = test_dataset[len_fitted_scene].near_far
    ndc_ray = args.ndc_ray

    # init resolution
    # upsamp_list = args.upsamp_list
    # update_AlphaMask_list = args.update_AlphaMask_list if args.update_AlphaMask_list is not None else []
    # n_lamb_sigma = args.n_lamb_sigma
    # n_lamb_sh = args.n_lamb_sh

    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}_{len_fitted_scene}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/distill/imgs_vis', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    aabb = train_dataset[len_fitted_scene].scene_bbox.to(device)
    # reso_cur = N_to_reso(args.N_voxel_init, aabb)
    # nSamples = min(args.nSamples, cal_n_samples(reso_cur,
    #                                             args.step_ratio))  # sqrt(128**2+128**2+128**2)/step(0.5)=443, nSample never more than 443 samples for one pixel
    if len_fitted_scene!=0 and len(args.sc_ckpt_list) != len_fitted_scene:
        print('args.sc_ckpt_list doesnt match args.datadir_list. CHECK!')
        return
    tensorf_teas = []
    if len_fitted_scene!= 0:
        for ckpt_name in args.sc_ckpt_list:
            ckpt = torch.load(ckpt_name, map_location=device)
            kwargs = ckpt['kwargs']
            kwargs.update({'device': device})
            tensorf_tea = eval(args.model_name+"_Distill")(**kwargs)
            tensorf_tea.load(ckpt)
            tensorf_teas.append(tensorf_tea)

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf_tea = eval(args.model_name+"_Distill")(**kwargs)
    tensorf_tea.load(ckpt)
    tensorf_teas.append(tensorf_tea)

    if args.student_ckpt is not None :
        if not os.path.exists(args.student_ckpt):
            print('the student ckpt path does not exists!! ')
            return
        stu_ckpt = torch.load(args.student_ckpt, map_location=device)
        stu_args = stu_ckpt['kwargs']
        stu_args.update({'device': device})
        stu_args.update({'scene_num': len_fitted_scene + 1})
    else:
        if len_fitted_scene > 0:
            print('Errpr ,fitted scene > 1,however no ckpt loaded.SET stu_ckpt path first. This will learn from scratch')
            # exit()
        print('Distill the student model from scratch!!!!')
        # Vanilla NeRF
        # stu_args = {'distance_scale':tensorf_tea.distance_scale,'rayMarch_weight_thres':ckpt['kwargs']['rayMarch_weight_thres'],\
        #         'aabb':tensorf_tea.aabb,'gridSize':tensorf_tea.gridSize,'near_far' : tensorf_tea.near_far
        #         ,'density_shift':tensorf_tea.density_shift,'step_ratio':tensorf_tea.step_ratio}
        # HypernetVanilla NeRF
        stu_args = {'distance_scale':tensorf_tea.distance_scale,'rayMarch_weight_thres':ckpt['kwargs']['rayMarch_weight_thres'],\
                'aabb':tensorf_tea.aabb,'gridSize':tensorf_tea.gridSize,'near_far' : tensorf_tea.near_far
                ,'density_shift':tensorf_tea.density_shift,'step_ratio':tensorf_tea.step_ratio,'device':device
                ,'pos_pe':args.dis_network_pos_pe,'dir_pe':args.dis_network_dir_pe,'z_dim':args.dis_network_z_dim, 'c_dim':args.dis_network_c_dim,'scene_num':(len_fitted_scene + 1) }
    stu_model = eval(args.stu_model_name)(**stu_args)
    if len_fitted_scene > 0 and args.student_ckpt is None:
        print('fitted scene > 1 ,so you must indicate one args.student_ckpt,This will learn from scratch')
        # exit()
    elif args.student_ckpt is not None :
        stu_model.load(stu_ckpt)

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
    grad_var_coff_rgbs = nn.Parameter(torch.tensor([0.045],device=device),requires_grad=True)
    grad_var_coff_alphas = nn.Parameter(torch.tensor([0.06],device=device),requires_grad=True)
    grad_var_coff = [{'params':grad_var_coff_rgbs, 'lr':args.dis_rgb_and_alpha_coff_lr_init}] + [{'params':grad_var_coff_alphas, 'lr':args.dis_rgb_and_alpha_coff_lr_init}]
    grad_vars += grad_var_coff
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logrithmic space
    # N_voxel_list = (torch.round(torch.exp(
    #     torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[
    #                1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]


    # if not args.ndc_ray:
    #     allrays, allrgbs = tensorf_tea.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(train_dataset[len_fitted_scene].all_rays.shape[0], args.batch_size)


    pbar = tqdm(range(args.dis_n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:

        if iteration < args.sc_switch_warmup_iter:
            scene_id = len_fitted_scene
        else:
            scene_id = int(( iteration / args.sc_switch_iter ) % (len_fitted_scene + 1))
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = train_dataset[scene_id].all_rays[ray_idx].to(device), train_dataset[scene_id].all_rgbs[ray_idx].to(device)

        # data example:
        # rays_train [4096,6], rgb_train [4096,3], rgb_maps [4096,3] depth_maps [4096]
        with torch.no_grad():
            tea_rgb_maps, tea_depth_maps, tea_rgbs, tea_sigmas, tea_alphas, tea_density_feats, tea_app_feats, xyz_sampled,viewdirs , z_vals, ray_valid  = tea_renderer(rays_train, tensorf_teas[scene_id], chunk=args.batch_size,
                                                                            N_samples=-1, white_bg=white_bg,
                                                                            ndc_ray=ndc_ray, device=device, is_train=False)

            #e.g. rgb_maps [4096,3];depth_maps [4096]; rgbs [4096,443,3]; sigmas,alphas: [4096,443]
            # app_feats [19817,27]->youwenti xyz_sampled [4096,443,3]; z_vals [4096,443]; ray_valid [4096,443]
        stu_rgb_maps, stu_depth_maps, stu_rgbs, stu_sigmas, stu_alphas, _, stu_app_feats = stu_renderer(stu_model, rays_train, xyz_sampled, viewdirs, z_vals, ray_valid,  chunk=args.dis_batch_size,  ndc_ray=ndc_ray, white_bg = white_bg, is_train=True, device = device,scene_id = scene_id)
        # test whether switch model right
        # img = (rgb_train.cpu().numpy().reshape(16, 8, 3) * 255).astype('uint8')
        # img = cv2.resize(img,(320,640))
        # imageio.imwrite('/home/yuze/Downloads/GT_{}.jpg'.format(iteration),img)
        # img = (tea_rgb_maps.cpu().numpy().reshape(16, 8, 3) * 255).astype('uint8')
        # img = cv2.resize(img,(320,640))
        # imageio.imwrite('/home/yuze/Downloads/tea_rgb_maps_{}.jpg'.format(iteration),img)
        # img = (stu_rgb_maps.detach().cpu().numpy().reshape(16, 8, 3) * 255).astype('uint8')
        # img = cv2.resize(img,(320,640))
        # imageio.imwrite('/home/yuze/Downloads/stu_rgb_maps_{}.jpg'.format(iteration),img)
        # loss
        total_loss = 0
        # cauculate loss
        if (iteration + 1 >= loss_hyperparam['dis_start_appfeatloss_iter']) and  (iteration + 1 < loss_hyperparam['dis_end_appfeatloss_iter']):
            assert(tea_app_feats.shape == stu_app_feats.shape),'app_feat size dont match between student and teacher'
            appfeatloss = loss_hyperparam['dis_appfeatloss_weight'] * torch.mean((tea_app_feats - stu_app_feats)**2)
            total_loss += appfeatloss
            summary_writer.add_scalar('train/appfeatloss_scene{}'.format(scene_id), appfeatloss.detach().item(), global_step=iteration)
        if (iteration + 1 >= loss_hyperparam['dis_start_rfloss_iter']) and  (iteration + 1 < loss_hyperparam['dis_end_rfloss_iter']):
            assert (tea_sigmas.shape == stu_sigmas.shape) and (tea_rgbs.shape == stu_rgbs.shape), 'app_feat size dont match between student and teacher'
            # rfloss = loss_hyperparam['dis_rfloss_weight'] * (1/(2*grad_var_coff_alphas*grad_var_coff_alphas)*torch.mean((tea_alphas[ray_valid] - stu_alphas[ray_valid]) ** 2) + \
            #                                                  1/(2*grad_var_coff_rgbs*grad_var_coff_rgbs)* torch.mean((tea_rgbs[ray_valid] - stu_rgbs[ray_valid]) **2) + torch.log(grad_var_coff_alphas * grad_var_coff_rgbs) )
            rfloss = torch.mean((tea_alphas[ray_valid] - stu_alphas[ray_valid]) ** 2) + 1.5 * torch.mean(
                (tea_rgbs[ray_valid] - stu_rgbs[ray_valid]) ** 2)
            # if torch.rand(1) < (iteration/args.dis_n_iters) +0.45:
            #     rfloss = torch.mean((tea_alphas[rgb_train.sum(dim=-1) <2.9] - stu_alphas[rgb_train.sum(dim=-1) <2.9]) **2) + 1.5* torch.mean((tea_rgbs[rgb_train.sum(dim=-1) != 3] - stu_rgbs[rgb_train.sum(dim=-1) != 3]) **2)
            # else:
            #     rfloss = torch.mean((tea_alphas[ray_valid] - stu_alphas[ray_valid]) ** 2) + 1.5* torch.mean((tea_rgbs[ray_valid] - stu_rgbs[ray_valid]) **2)
            total_loss += (1/3)*rfloss
            summary_writer.add_scalar('train/rfloss_scene{}'.format(scene_id), rfloss.detach().item(), global_step=iteration)
            summary_writer.add_scalar('train/grad_var_coff_rgbs_scene{}'.format(scene_id),grad_var_coff_rgbs.detach().item(),global_step=iteration)
            summary_writer.add_scalar('train/grad_var_coff_alphas_scene{}'.format(scene_id), grad_var_coff_alphas.detach().item(),global_step=iteration)
        if (iteration + 1 >= loss_hyperparam['dis_start_ftloss_iter']) and  (iteration + 1 < loss_hyperparam['dis_end_ftloss_iter']):
            assert (tea_rgb_maps.shape == stu_rgb_maps.shape), 'app_feat size dont match between student and teacher'
            # if torch.rand(1) < (iteration / args.dis_n_iters)  +0.45:
            #     ftloss = loss_hyperparam['dis_ftloss_weight'] * torch.mean((stu_rgb_maps[rgb_train.sum(dim=-1) != 3] - rgb_train[rgb_train.sum(dim=-1) != 3]) ** 2)
            # else:
            #     ftloss = loss_hyperparam['dis_ftloss_weight'] * torch.mean((stu_rgb_maps - rgb_train) ** 2)
            ftloss = loss_hyperparam['dis_ftloss_weight'] * torch.mean((stu_rgb_maps - rgb_train) ** 2)
            total_loss +=  ftloss
            summary_writer.add_scalar('train/ftloss_scene{}'.format(scene_id), ftloss.detach().item(), global_step=iteration)
        summary_writer.add_scalar('train/total_loss_scene{}'.format(scene_id), total_loss.detach().item(), global_step=iteration)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        mse = torch.mean((stu_rgb_maps - tea_rgb_maps) ** 2).detach().item()

        PSNRs.append(-10.0 * np.log(mse) / np.log(10.0))

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor


        if iteration in [loss_hyperparam['dis_end_appfeatloss_iter'], loss_hyperparam['dis_end_rfloss_iter'],loss_hyperparam['dis_end_ftloss_iter']]:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
                grad_vars = stu_model.get_optparam_groups(args.dis_lr_init)
                optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        # for name, param in stu_model.density_linear.named_parameters():
        #     summary_writer.add_scalar(f"sdf_gradient/{name}_grad_norm_density_linear", torch.norm(param.grad), iteration)
        # for name, param in stu_model.encoder.named_parameters():
        #     summary_writer.add_scalar(f"sdf_gradient/{name}_grad_norm_encoder", torch.norm(param.grad), iteration)
        # for name, param in stu_model.app_linear.named_parameters():
        #     summary_writer.add_scalar(f"sdf_gradient/{name}_grad_norm_app_linear", torch.norm(param.grad), iteration)

        summary_writer.add_scalar('train/PSNR_scene{}'.format(scene_id), PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse_scene{}'.format(scene_id), mse, global_step=iteration)
        summary_writer.add_scalar('train/curr_loss_scene{}'.format(scene_id), total_loss.detach().item(), global_step=iteration)
        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f'current_scene_id: [{scene_id:02d}]'
                + f' curr_loss = {float(total_loss.detach().item()):.2f}'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {mse:.6f}'
            )
            PSNRs = []

        if iteration % args.dis_vis_every == args.dis_vis_every - 1 and args.dis_N_vis != 0:
            stu_model.save(f'{logfolder}/distill_{args.expname}_{iteration}.th')
            PSNRs_test = evaluation_student_model(test_dataset, stu_model, args, stu_renderer, f'{logfolder}/distill/imgs_vis/', N_vis=args.dis_N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
    stu_model.save(f'{logfolder}/distill_{args.expname}.th')

    if args.dis_render_train:
        os.makedirs(f'{logfolder}/distill/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_student_model(train_dataset, stu_model, args, stu_renderer, f'{logfolder}/distill/img_train_all/', N_vis=-1,
                                     N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/distill/img_test_all', exist_ok=True)
        test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation_student_model(test_dataset, stu_model, args, stu_renderer,
                                              f'{logfolder}/distill/img_test_all/', N_vis=-1,
                                              N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                                              compute_extra_metrics=False)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=args.dis_n_iters)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')
@torch.no_grad()
def test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf_tea = eval(args.model_name)(**kwargs)
    tensorf_tea.load(ckpt)

    if args.student_ckpt is not None:
        if not os.path.exists(args.student_ckpt):
            print('the student ckpt path does not exists!! ')
            return
        print(args.student_ckpt)
        stu_ckpt = torch.load(args.student_ckpt, map_location=device)
        stu_args = stu_ckpt['kwargs']
        stu_args.update({'device': device})
    else:
        print('Set the student ckpt, plz!! ')
        exit()
    stu_model = eval(args.stu_model_name)(**stu_args)
    stu_model.load(stu_ckpt)



    logfolder = f'{args.basedir}/{args.expname}'
    if args.render_train:
        os.makedirs(f'{logfolder}/distill/imgs_train_all', exist_ok=True)
        train_dataset = [dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)]
        PSNRs_test = evaluation_student_model(train_dataset, stu_model, args, stu_renderer, f'{logfolder}/distill/img_train_all/', N_vis=-1,
                                     N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/distill/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation_student_model(test_dataset, stu_model, args, stu_renderer, f'{logfolder}/distill/img_train_all/', N_vis=-1,
                                     N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)




if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20230601)
    np.random.seed(20230601)

    args = config_parser()
    print(args)

    if args.dis_reconstruction:
         distill(args)

