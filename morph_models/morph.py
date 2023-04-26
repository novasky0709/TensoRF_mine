from opt import config_parser
from renderer import *
from torch.utils.tensorboard import SummaryWriter
import datetime
from dataLoader import dataset_dict
from morph_models.piggy_models.piggyBackTesnorVMSplit import PiggyBackTensorVMSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renderer = OctreeRender_trilinear_fast

class SimpleSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]


def OctreeRender_trilinear_fast(rays, piggyBack_model, chunk=4096, N_samples=-1, ndc_ray=False, white_bg=True, is_train=False,
                                device='cuda'):
    rgbs, alphas, depth_maps, weights, uncertainties = [], [], [], [], []
    N_rays_all = rays.shape[0]
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        rgb_map, depth_map, rgb, sigma, alpha, weight, bg_weight = piggyBack_model(rays_chunk, is_train=is_train, white_bg=white_bg, ndc_ray=ndc_ray,
                                                    N_samples=N_samples)


        rgbs.append(rgb_map)
        depth_maps.append(depth_map)
    return torch.cat(rgbs), None, torch.cat(depth_maps), None, None
@torch.no_grad()
def evaluation(test_dataset,piggyBack_model, args, renderer, savePath=None, N_vis=5, prtx='', N_samples=-1,
               white_bg=False, ndc_ray=False, compute_extra_metrics=True, device='cuda'):

    PSNRs, rgb_maps, depth_maps = [], [], []
    ssims,l_alex,l_vgg=[],[],[]
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath+"/rgbd", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis,1)
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval))
    for idx, samples in tqdm(enumerate(test_dataset.all_rays[0::img_eval_interval]), file=sys.stdout):

        W, H = test_dataset.img_wh
        rays = samples.view(-1,samples.shape[-1])

        rgb_map, _, depth_map, _, _ = renderer(rays, piggyBack_model, chunk=4096, N_samples=N_samples,
                                        ndc_ray=ndc_ray, white_bg = white_bg, device=device)
        rgb_map = rgb_map.clamp(0.0, 1.0)

        rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(),near_far)
        if len(test_dataset.all_rgbs):
            gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
            loss = torch.mean((rgb_map - gt_rgb) ** 2)
            PSNRs.append(-10.0 * np.log(loss.item()) / np.log(10.0))

            if compute_extra_metrics:
                ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', piggyBack_model.device)
                l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', piggyBack_model.device)
                ssims.append(ssim)
                l_alex.append(l_a)
                l_vgg.append(l_v)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)
        if savePath is not None:
            imageio.imwrite(f'{savePath}/{prtx}{idx:03d}.png', rgb_map)
            rgb_map = np.concatenate((rgb_map, depth_map), axis=1)
            imageio.imwrite(f'{savePath}/rgbd/{prtx}{idx:03d}.png', rgb_map)

    imageio.mimwrite(f'{savePath}/{prtx}video.mp4', np.stack(rgb_maps), fps=30, quality=10)
    imageio.mimwrite(f'{savePath}/{prtx}depthvideo.mp4', np.stack(depth_maps), fps=30, quality=10)

    if PSNRs:
        psnr = np.mean(np.asarray(PSNRs))
        if compute_extra_metrics:
            ssim = np.mean(np.asarray(ssims))
            l_a = np.mean(np.asarray(l_alex))
            l_v = np.mean(np.asarray(l_vgg))
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr, ssim, l_a, l_v]))
        else:
            np.savetxt(f'{savePath}/{prtx}mean.txt', np.asarray([psnr]))


    return PSNRs



def morph_distill(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=False,N_vis = -1)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, is_stack=True,N_vis = 10)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray
    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list if args.update_AlphaMask_list is not None else []
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    # init log file
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'

    # init log file
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    os.makedirs(f'{logfolder}/imgs_rgba', exist_ok=True)
    os.makedirs(f'{logfolder}/rgba', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    # init parameters
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))#sqrt(128**2+128**2+128**2)/step(0.5)=443, nSample never more than 443 samples for one pixel

    if not (os.path.exists(args.morph_src_ckpt) or os.path.exists(args.morph_dst_ckpt)):
        print('the src or dst ckpt path does not exists!! pretrain first')
        return
    # load model
    src_ckpt = torch.load(args.morph_src_ckpt, map_location=device)
    dst_ckpt = torch.load(args.morph_dst_ckpt, map_location=device)
    dst_kwargs = dst_ckpt['kwargs']
    dst_kwargs.update({'device': device})
    src_kwargs = src_ckpt['kwargs']
    src_kwargs.update({'device': device})
    dst_tensorf = eval(args.model_name)(**dst_kwargs)
    dst_tensorf.load(dst_ckpt)
    src_tensorf = eval(args.model_name)(**src_kwargs)
    src_tensorf.load(src_ckpt)
    # init morph model
    # assume the same kwargs of src_ckpt and dst_ckpt
    piggyBack_model = eval(args.morph_model_name)(src_kwargs,dst_kwargs,tensorf_model = src_tensorf,train_strategy = 'distill',piggyback_num = 1,device = device)
    grad_vars = piggyBack_model.get_optparam_groups(args.m_lr_basis)
    if args.m_lr_decay_iters > 0 :
        lr_factor = args.m_lr_decay_target_ratio ** (1 / args.m_lr_decay_iters)
    else:
        args.m_lr_decay_iters = args.m_n_iters
        lr_factor = args.m_lr_decay_target_ratio ** (1 / args.m_n_iters)
    print("morph lr decay", args.m_lr_decay_target_ratio, args.m_lr_decay_iters)
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]


    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = piggyBack_model.tensorf_model.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = SimpleSampler(allrays.shape[0], 256) # args.batch_size
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)

    for iteration in pbar:

        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx].to(device), allrgbs[ray_idx].to(device)
        # todo: 这里都做成chunk
        target_rgb_map, target_depth_map , target_rgb, target_sigma, target_alpha, target_weight, target_bg_weight = dst_tensorf(rays_train)
        rgb_map, depth_map , rgb, sigma, alpha, weight, bg_weight = piggyBack_model(rays_train)

        # rgb_map, alphas_map, depth_map, weights, uncertainty

        distill_loss = torch.mean((torch.cat([sigma,rgb],dim = -1) - torch.cat([target_sigma[...,None],target_rgb],dim = -1)) ** 2)
        rgb_loss = torch.mean((target_rgb_map - rgb_map) ** 2)
        depth_loss = torch.mean((target_depth_map - depth_map) ** 2)
        total_loss = 0*distill_loss + rgb_loss + 0*depth_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = rgb_loss.detach().item()

        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []

        if iteration % args.m_vis_every == args.m_vis_every - 1 and args.N_vis != 0:
            PSNRs_test = evaluation(test_dataset, piggyBack_model, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg=white_bg, ndc_ray=ndc_ray,
                                    compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

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
        #
        #     if args.lr_upsample_reset:
        #         print("reset lr to initial")
        #         lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
        #     else:
        #         lr_scale = args.lr_decay_target_ratio ** (iteration / args.m_n_iters)
        #     grad_vars = tensorf.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale)
        #     optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    piggyBack_model.save(f'{logfolder}/{args.expname}.th')

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        # c2ws = test_dataset.poses
        print('========>', c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg=white_bg, ndc_ray=ndc_ray, device=device)
def morph_finetune(args):
    pass
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)
    args = config_parser()
    print(args)
    if args.morph == 1:
        morph_distill(args)
    elif args.morph == 2:
        morph_finetune(args)