import os
import yaml
import torch
import xatlas
import time
import argparse
import json

import numpy as np
import nvdiffrast.torch as dr

from render import mesh
from render import material
from render import texture
from render import render
from render import light
from render import util
from render import obj
from render import mlptexture

import render.renderutils as ru
from dataset.dataset_brdf import DatasetBRDF
from render.dlmesh import DLMesh

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')   #创建一个与输入图像具有相同分辨率但通道数为3的PyTorch张量
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background

    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)  #torch.lerp 用于在背景图像和前三个通道的图像之间进行线性插值。插值的权重由 target['img'][..., 3:4] 给出，通常表示图像的 alpha 通道
    
    return target

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)

    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)

    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])

    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)
    
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'  : mat['bsdf'],
        'kd'    : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'    : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal': texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it)

def optimize_mesh(
    glctx,
    geometry,
    opt_material,
    lgt,
    dataset_train,
    dataset_validate,
    FLAGS,
    warmup_iter=0,
    log_interval=10,
    pass_idx=0,
    pass_name="",
    optimize_light = True,
    optimize_geometry=True
    ):
    
    # ==============================================================================================
    #  Setup torch optimizer
    # ==============================================================================================
 
    learning_rate = FLAGS.learning_rate[pass_idx] if isinstance(FLAGS.learning_rate, list) or isinstance(FLAGS.learning_rate, tuple) else FLAGS.learning_rate   #列表或元组类型
    learning_rate_pos = learning_rate[0] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate
    learning_rate_mat = learning_rate[1] if isinstance(learning_rate, list) or isinstance(learning_rate, tuple) else learning_rate

    def lr_schedule(iter, fraction):
        if iter < warmup_iter:
            return iter / warmup_iter
        return max(0.0, 10**(-(iter - warmup_iter)*0.0002))  # Exponential falloff from [1.0, 0.1] over 5k epochs. 在 5000 个迭代周期内，从 [1.0, 0.1] 指数下降
    
    # ==============================================================================================
    #  Image loss
    # ==============================================================================================
    image_loss_fn = createLoss(FLAGS)

    trainer_noddp = Trainer(glctx, geometry, lgt, opt_material, optimize_geometry, optimize_light, image_loss_fn, FLAGS)

    if FLAGS.multi_gpu:     # Multi GPU training mode
        import apex
        from apex.parallel import DistributedDataParallel as DDP

        trainer = DDP(trainer_noddp)
        trainer.train()
        if optimize_geometry:
            optimizer_mesh = apex.optimizers.FusedAdam(trainer_noddp.geo_params, lr=learning_rate_pos)  #优化器
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9)) #学习率调整器

        optimizer = apex.optimizers.FusedAdam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))
    else:   # Single GPU training mode
        trainer = trainer_noddp
        if optimize_geometry:
            optimizer_mesh = torch.optim.Adam(trainer_noddp.geo_params, lr=learning_rate_pos)
            scheduler_mesh = torch.optim.lr_scheduler.LambdaLR(optimizer_mesh, lr_lambda=lambda x: lr_schedule(x, 0.9))

        optimizer = torch.optim.Adam(trainer_noddp.params, lr=learning_rate_mat)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_schedule(x, 0.9))

    # ==============================================================================================
    #  Training loop
    # ==============================================================================================
    img_cnt = 0
    img_loss_vec = []
    reg_loss_vec = []
    iter_dur_vec = []

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=FLAGS.batch, collate_fn=dataset_train.collate, shuffle=True)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_train.collate)

    def cycle(iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)    #如果迭代器还有下一个元素，就通过 yield 关键字产生这个元素，然后继续下一次循环
            except StopIteration:
                iterator = iter(iterable)

    v_it = cycle(dataloader_validate)

    for it, target in enumerate(dataloader_train):
        target = prepare_batch(target, 'random')    # Mix randomized background into dataset image

        # ==============================================================================================
        #  Display / save outputs. Do it before training so we get initial meshes
        # ==============================================================================================

        # Show/save image before training step (want to get correct rendering of input)
        if FLAGS.local_rank == 0:
            display_image = FLAGS.display_interval and (it % FLAGS.display_interval == 0)
            save_image = FLAGS.save_interval and (it % FLAGS.save_interval == 0)
            if display_image or save_image:
                result_image, result_dict = validate_itr(glctx, prepare_batch(next(v_it), FLAGS.background), geometry, opt_material, lgt, FLAGS)
                np_result_image = result_image.detach().cpu().numpy()
                if display_image:
                    util.display_image(np_result_image, title='%d / %d' % (it, FLAGS.iter))
                if save_image:
                    util.save_image(FLAGS.out_dir + '/' + ('img_%s_%06d.png' % (pass_name, img_cnt)), np_result_image)      # 06d:数字的总宽度为 6 位。如果数字的位数不足 6 位，将用零填充
                    img_cnt = img_cnt+1
                    
        iter_start_time = time.time()

        # ==============================================================================================
        #  Zero gradients
        # ==============================================================================================
        optimizer.zero_grad()
        if optimize_geometry:
            optimizer_mesh.zero_grad()

        # ==============================================================================================
        #  Training
        # ==============================================================================================
        img_loss, reg_loss = trainer(target, it)

        # ==============================================================================================
        #  Final loss
        # ==============================================================================================
        total_loss = img_loss + reg_loss

        img_loss_vec.append(img_loss.item())    #item() 是 PyTorch 张量的方法，它用于将张量中的数值提取为 Python 基本数据类型
        reg_loss_vec.append(reg_loss.item())

        # ==============================================================================================
        #  Backpropagate
        # ==============================================================================================
        total_loss.backward()
        if hasattr(lgt, 'base') and lgt.base.grad is not None and optimize_light:   #hasattr(lgt, 'base')：这个条件检查变量 lgt 是否具有名为 'base' 的属性
            lgt.base.grad *=64
        if 'kd_ks_normal' in opt_material:
            opt_material['kd_ks_normal'].encoder.params.grad /= 8.0

        optimizer.step()    #执行参数更新
        scheduler.step()    #调整学习率

        if optimize_geometry:
            optimizer_mesh.step()
            scheduler_mesh.step()

        # ==============================================================================================
        #  Clamp trainables to reasonable range  将可训练参数限制在合理范围内
        # ==============================================================================================
        with torch.no_grad():
            if 'kd' in opt_material:
                opt_material['kd'].clamp_()     # _ 表示原地操作（In-Place Operation）。原地操作是指该方法会直接修改调用它的对象，而不会创建新的对象或副本
            if 'ks' in opt_material:
                opt_material['ks'].clamp_()
            if 'normal' in opt_material:
                opt_material['normal'].clamp_()
                opt_material['normal'].normalize_()
            if lgt is not None:
                lgt.clamp_(min=0.0)

        torch.cuda.current_stream().synchronize()
        iter_dur_vec.append(time.time() - iter_start_time)

        # ==============================================================================================
        #  Logging
        # ==============================================================================================
        if it % log_interval == 0 and FLAGS.local_rank == 0:
            img_loss_avg = np.mean(np.asarray(img_loss_vec[-log_interval:]))    #计算最新 log_interval 个迭代中图像损失的均值。img_loss_vec 是一个存储每个迭代图像损失的列表，[-log_interval:] 表示取列表的最后 log_interval 个元素
            reg_loss_avg = np.mean(np.asarray(reg_loss_vec[-log_interval:]))
            iter_dur_avg = np.mean(np.asarray(iter_dur_vec[-log_interval:]))

            remaining_time = (FLAGS.iter - it) * iter_dur_avg
            print("iter=%5d, img_loss=%.6f, reg_loss=%.6f, lr=%.5f, time=%.1f ms, rem=%s" % (it, img_loss_avg, reg_loss_avg, optimizer.param_groups[0]['lr'], iter_dur_avg*1000, util.time_to_text(remaining_time)))
            # 迭代次数（5字符整数）, 图像损失（6 位小数浮点数）, 正则化损失, 学习率, 迭代时间, 剩余时间
    return geometry, opt_material

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        if FLAGS.display is not None:
            white_bg = torch.ones_like(target['background'])
            for layer in FLAGS.display:
                if 'latlong' in layer and layer['latlong']:
                    if isinstance(lgt, light.EnvironmentLight):
                        result_dict['light_image'] = util.cubemap_to_latlong(lgt.base, FLAGS.display_res)
                    result_image = torch.cat([result_image, result_dict['light_image']], axis=1)
                elif 'relight' in layer:
                    if not isinstance(layer['relight'], light.EnvironmentLight):
                        layer['relight'] = light.load_env(layer['relight'])
                    img = geometry.render(glctx, target, layer['relight'], opt_material)
                    result_dict['relight'] = util.rgb_to_srgb(img[..., 0:3])[0]
                    result_image = torch.cat([result_image, result_dict['relight']], axis=1)
                elif 'bsdf' in layer:
                    buffers = geometry.render(glctx, target, lgt, opt_material, bsdf=layer['bsdf'])
                    if layer['bsdf'] == 'kd':
                        result_dict[layer['bsdf']] = util.rgb_to_srgb(buffers['shaded'][0, ..., 0:3])
                    elif layer['bsdf'] == 'normal':
                        result_dict[layer['bsdf']] = (buffers['shaded'][0, ..., 0:3] + 1) * 0.5
                    else:
                        result_dict[layer['bsdf']] = buffers['shaded'][0, ..., 0:3]
                    result_image = torch.cat([result_image, result_dict[layer['bsdf']]], axis=1)

        return result_image, result_dict

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):
    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    compute_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device='cuda')
    compute_lpips = LearnedPerceptualImagePatchSimilarity().to(device='cuda')

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)  # 从 dataset_validate 中逐个加载样本，并使用 collate 函数对批次数据进行整理

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrices.txt'), 'w') as fout:
        #fout.write('ID, MSE, PSNR\n')
        fout.write('ID, MSE, PSNR, SSIM, LPIPS\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate): 
            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)

            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0)
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()   #计算均方误差（Mean Squared Error，MSE）损失，并将结果提取为一个标量值
            mse_values.append(float(mse))
            psnr=util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            ssim = compute_ssim(opt[None, ...].permute(0, 3, 1, 2).contiguous(), ref[None, ...].permute(0, 3, 1, 2).contiguous())
            ssim_values.append(float(ssim))
            lpips = compute_lpips(opt[None, ...].permute(0, 3, 1, 2).contiguous(), ref[None, ...].permute(0, 3, 1, 2).contiguous())
            lpips_values.append(float(lpips))

            #line = "%d, %1.8f,  %1.8f\n" % (it, mse, psnr)  #   %d：整数占位符; %1.8f：浮点数占位符，保留小数点后 8 位
            line = "%d, %1.8f,  %1.8f,  %1.8f,  %1.8f\n" % (it, mse, psnr, ssim, lpips)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()  #从 result_dict 字典中获取键 k 对应的值,通过 detach() 方法将张量从计算图中分离，然后通过 cpu().numpy() 方法将其转换为 NumPy 数组
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)  # %06d 表示将一个整数 it 格式化为 6 位宽度的数字

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        avg_ssim = np.mean(np.array(ssim_values))
        avg_lpips = np.mean(np.array(lpips_values))
        # line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        # fout.write(str(line))
        # print("MSE,      PSNR")
        # print("%1.8f, %2.3f" % (avg_mse, avg_psnr))

        line = "AVERAGES: %1.4f, %2.3f, %2.3f, %2.3f\n" % (avg_mse, avg_psnr, avg_ssim, avg_lpips)
        fout.write(str(line))
        print("MSE,    PSNR,    SSIM,    LPIPS")
        print("%1.8f, %2.3f, %2.3f, %2.3f" % (avg_mse, avg_psnr, avg_ssim, avg_lpips))

    return avg_psnr, avg_ssim, avg_lpips

########################    main     ###############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    # argparse 是 Python 的一个标准库，用于处理命令行参数和选项, ArgumentParser 是 argparse 库中的一个类，用于创建和配置命令行参数解析器。description='nvdiffrec' 是 ArgumentParser 类的一个参数，用于设置解析器的描述信息
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])     # nargs=2 是 argparse 库中用于参数定义的一个选项，它指示命令行参数解析器应该期望接受两个参数值作为一个单一的命令行参数
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)

    FLAGS = parser.parse_args()
    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier 环境贴图强度倍增器
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = True                     # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'     #指定主机地址的环境变量
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'         #指定主机端口的环境变量

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")  # nccl是由NVIDIA开发的一个高性能的GPU加速通信库

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]
        
    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()
    
    if os.path.isdir(FLAGS.ref_mesh):
        if os.path.isfile(os.path.join(FLAGS.ref_mesh, 'metadata.yaml')):
            dataset_train = DatasetBRDF(os.path.join(FLAGS.ref_mesh, 'metadata.yaml'), FLAGS, examples=(FLAGS.iter + 1)*FLAGS.batch)
            dataset_validate = DatasetBRDF(os.path.join(FLAGS.ref_mesh, 'metadata_val.yaml'), FLAGS)
        else:
            print("Warnning: yaml file error!")
    else:
            print("Warnning: ref_mesh path error!")
    
    # 导入mesh
    base_mesh = mesh.load_mesh(FLAGS.base_mesh)
    yaml_file = os.path.join(FLAGS.ref_mesh, 'metadata.yaml')
    cfg = yaml.load(open(yaml_file, 'r'), Loader=yaml.FullLoader)
    xform_bounds = cfg['model_matrix']
    xform_bounds = torch.tensor(xform_bounds, dtype=torch.float32, device='cuda')

    v_pos = ru.xfm_points(base_mesh.v_pos[None, ...], xform_bounds[None, ...])
    base_mesh.v_pos = v_pos.squeeze(0)[:, :3].contiguous()

    geometry = DLMesh(base_mesh, FLAGS)
    # 纹理贴图
    mat = initial_guess_material(geometry, False, FLAGS, init_mat=base_mesh.material)

    # 光照贴图
    if FLAGS.learn_light:
        lgt = light.create_trainable_env_rnd(512, scale=0.0, bias=0.5)
    else:
        lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    # lgt.xfm(xform_bounds) 
    # lgt.build_mips()
    
    # 渲染&优化
    geometry, mat = optimize_mesh(glctx, geometry, mat, lgt, dataset_train, dataset_validate, FLAGS, pass_idx=0, pass_name="mesh_pass", 
                                            warmup_iter=100, optimize_light=not FLAGS.lock_light, optimize_geometry=FLAGS.lock_pos)
    
    # 验证
    if FLAGS.validate and FLAGS.local_rank == 0:
        validate(glctx, geometry, mat, lgt, dataset_validate, os.path.join(FLAGS.out_dir, "validate"), FLAGS)

    # 输出
    if FLAGS.local_rank == 0:
        final_mesh = geometry.getMesh(mat)
        os.makedirs(os.path.join(FLAGS.out_dir, "mesh"), exist_ok=True)
        obj.write_obj(os.path.join(FLAGS.out_dir, "mesh/"), final_mesh)
        light.save_env_map(os.path.join(FLAGS.out_dir, "mesh/probe.hdr"), lgt)