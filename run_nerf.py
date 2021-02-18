from load_blender import load_blender_data, load_blender_data_fill_up
from load_deepvoxels import load_dv_data
from load_llff import load_llff_data
from load_lightstage import load_lightstage_data
from run_nerf_helpers import *
import time
import random
import json
import imageio
import numpy as np
import tensorflow as tf
import sys
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


tf.compat.v1.enable_eager_execution()


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


# def run_network(inputs, viewdirs, sh, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    # embed view direction
    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

    # embed spherical harmonics
    # if sh is not None:
    #     sh = sh.reshape(-1)
    #     sh = tf.broadcast_to(sh, (embedded.shape[0], sh.shape[0]))
    #     embedded = tf.concat((embedded, sh), -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays(ray_batch,
                sh,
                light_probe,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                # network_material=None,
                white_bkgd=False,
                raw_noise_std=0.,
                ray_width=4.,

                network_fn_=None,
                network_query_fn_=None,
                N_samples_=None,
                retraw_=None,
                lindisp_=None,
                perturb_=None,
                N_importance_=None,
                network_fine_=None,
                white_bkgd_=None,
                raw_noise_std_=None,

                verbose=False
                ):
    """Volumetric rendering.

    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    def raw2outputs(raw, z_vals, rays_d, mask):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Extract albedo of each sample position along each ray.
        albedo = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

        norm = 2. * tf.math.sigmoid(raw[..., 4:7]) - 1.  # [N_rays, N_samples, 3]
        norm = norm / (tf.norm(norm, axis=2, keepdims=True) + 1e-6)  # [N_rays, N_samples, 3]

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * \
            tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        albedo_map = tf.reduce_sum(
            weights[..., None] * albedo, axis=-2)  # [N_rays, 3]
        norm_map = tf.reduce_sum(
            weights[..., None] * norm, axis=-2)  # [N_rays, 3]

        # Estimated depth map is expected distance.
        # depth_map = tf.reduce_sum(weights * z_vals, axis=-1)

        # Disparity map is inverse depth.
        # disp_map = 1./tf.maximum(1e-10, depth_map /
        #                          tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            albedo_map = albedo_map + (1.-acc_map[..., None])
            norm_map = norm_map + (1.-acc_map[..., None])

        # return rgb_map, albedo_map, sh_map, spec_map, sh_coef_out, disp_map, acc_map, weights, depth_map
        return albedo_map, norm_map, weights

    def raw2outputs_test(raw, raw_material, z_vals, rays_d, sh, light_probe, mask, albedo_gt, normal_gt):
        """Transforms model's predictions to semantically meaningful values.

        Args:
          raw: [num_rays, num_samples along ray, 4]. Prediction from model.
          z_vals: [num_rays, num_samples along ray]. Integration time.
          rays_d: [num_rays, 3]. Direction of each ray.

        Returns:
          rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
          disp_map: [num_rays]. Disparity map. Inverse of depth map.
          acc_map: [num_rays]. Sum of weights along each ray.
          weights: [num_rays, num_samples]. Weights assigned to each sampled color.
          depth_map: [num_rays]. Estimated distance to object.
        """
        # Function for computing density from model prediction. This value is
        # strictly between [0, 1].
        def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
            tf.exp(-act_fn(raw) * dists)

        # Compute 'distance' (in time) between each integration time along a ray.
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        # The 'distance' from the last integration time is infinity.
        dists = tf.concat(
            [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
            axis=-1)  # [N_rays, N_samples]

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point.
        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        # alpha_material = raw2alpha(raw_material[..., 3] + noise, dists)  # [N_rays, N_samples]
    
        # Extract albedo of each sample position along each ray.
        albedo = tf.broadcast_to(albedo_gt[:,None,:], alpha.shape.as_list() + [3])
        # norm = tf.broadcast_to(normal_gt[:,None,:], alpha.shape.as_list() + [3])

        # albedo = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        norm = 2. * tf.math.sigmoid(raw[..., 4:7]) - 1.  # [N_rays, N_samples, 3]

        spec = tf.math.sigmoid(raw[..., 7])  # [N_rays, N_samples,]
        lt_diffuse_pw = tf.nn.relu(raw[..., 8])  # [N_rays, N_samples,]
        lt_sh_pw = tf.nn.relu(raw[..., 9])  # [N_rays, N_samples,]
        lt_vis_diffuse = tf.nn.relu(raw[..., 10])  # [N_rays, N_samples,]
        lt_vis_sh = tf.nn.relu(raw[..., 11])  # [N_rays, N_samples,]

        spec = spec[..., None]
        lt_diffuse_pw = tf.reduce_mean(lt_diffuse_pw)
        lt_sh_pw = tf.reduce_mean(lt_sh_pw)
        lt_vis_diffuse = lt_vis_diffuse[..., None]
        lt_vis_sh = lt_vis_sh[..., None]

        norm = norm / (tf.norm(norm, axis=2, keepdims=True) + 1e-6)  # [N_rays, N_samples, 3]
        norm_x, norm_y, norm_z = tf.unstack(norm, axis=2)
        norm_x2, norm_y2, norm_z2 = norm_x*norm_x, norm_y*norm_y, norm_z*norm_z

        # 2021/02/03
        # try to stop gradient
        # Log: *_stop_grad
        # Commit: 
        # Results:
        # tf.stop_gradient(albedo)
        # tf.stop_gradient(norm)

        # Extract sphereical harmoncis coefficients
        sh_basis = [
            # level 0
            tf.cast(tf.broadcast_to(1.0 / 2.0 * np.sqrt(1.0 / np.pi), norm_x.shape.as_list()), tf.float32), # l = 0; m = 0
            # level 1
            np.sqrt(3.0 / (4.0 * np.pi)) * norm_y,  # l = 1; m = -1
            np.sqrt(3.0 / (4.0 * np.pi)) * norm_z,  # l = 1; m = 0
            np.sqrt(3.0 / (4.0 * np.pi)) * norm_x,  # l = 1; m = 1
            # level 2
            1.0 / 2.0 * np.sqrt(15.0 / np.pi) * norm_x * norm_y,
            1.0 / 2.0 * np.sqrt(15.0 / np.pi) * norm_z * norm_y,
            1.0 / 4.0 * np.sqrt(5.0 / np.pi)  * (-norm_x2-norm_y2 + 2.0*norm_z2),
            1.0 / 2.0 * np.sqrt(15.0 / np.pi) * norm_x * norm_z,
            1.0 / 4.0 * np.sqrt(15.0 / np.pi) * norm_x2 - norm_y2,
            # level 3
            1.0 / 4.0 * np.sqrt(35.0 / (2.0 * np.pi)) * (3.0 * norm_x2 - norm_y2) * norm_y,
            1.0 / 2.0 * np.sqrt(105.0 / np.pi)        * norm_x * norm_z * norm_y,
            1.0 / 4.0 * np.sqrt(21.0 / (2.0 * np.pi)) * norm_y * (5.0*norm_z2 - norm_x2 - norm_y2),
            1.0 / 4.0 * np.sqrt(7.0 / np.pi)          * norm_z * (1.5*norm_z2 - 3.0*norm_x2 - 3.0*norm_y2),
            1.0 / 4.0 * np.sqrt(21.0 / (2.0 * np.pi)) * norm_x * (5.0*norm_z2 - norm_x2 - norm_y2),
            1.0 / 4.0 * np.sqrt(105.0 / np.pi)        * (norm_x2 - norm_y2) * norm_z,
            1.0 / 4.0 * np.sqrt(35.0 / (2.0 * np.pi)) * (norm_x2 - 3.0*norm_y2) * norm_x
        ]
        sh_basis = tf.stack(sh_basis, axis=-1) # [N_rays, N_samples, 16]
        sh = tf.broadcast_to(sh, sh_basis.shape.as_list()[:2] + list(sh.shape)) # [N_rays, N_samples, 16, 3]
        lt_sh = tf.reduce_sum(sh * sh_basis[..., None], axis=-2) # [N_rays, N_samples, 3]

        # for direct light
        # down_step = 100
        # light_probe = light_probe[::down_step,::down_step,:] # [h,w,3]

        # get uv coordinates
        h, w, _ = light_probe.shape
        u = np.arange(w)/w
        v = np.arange(h)/h

        # get spherical coordinates
        r = 1
        rot = 0.25
        theta = v * np.pi # 0 to pi
        phi = (u+rot) * 2*np.pi # -pi to pi

        # spherical coordinates to cartesian coordinates
        X = r * np.sin(theta[..., None]) * np.cos(phi) # [h,w]
        Y = r * np.sin(theta[..., None]) * np.sin(phi) # [h,w]
        Z = r * np.cos(theta[..., None]) * np.ones(phi.shape) # [h,w]
        x = np.reshape(X, -1)
        y = np.reshape(Y, -1)
        z = np.reshape(Z, -1)
        l_dir = np.stack([x, y, z], axis=-1).astype(np.float32) # [h*w,3]
        l_weight = np.sin(theta) # [h,]
        l_color = np.reshape(light_probe * l_weight[:, None, None], (-1,3)).astype(np.float32) # [h*w,3]
        nDotL = tf.maximum(tf.matmul(norm, l_dir, transpose_b=True) / l_color.shape[0], 0.) # [N_rays, N_samples, 3] * [3, h*w] -> [N_rays, N_samples, h*w]
        lt_diffuse = tf.matmul(nDotL, l_color) # [N_rays, N_samples, h*w] * [h*w,3] -> [N_rays, N_samples, 3]

        # 2021/01/21
        # test if model_1_sh_* data works for my case
        # Log: *_baseline
        # Results: gt albedo and gt normal works
        # Commit: 7f48425c50eb66e08408525a6896fc857e20badf
            # rgb = lt_diffuse_pw * albedo * lt_diffuse + lt_sh_pw * lt_sh
            # loss = albedo_loss + normal_loss + img_loss0

        # 2021/01/22
        # let network pred visilibity map, which means to what extent can the sample be lighten up by the light point
        # which decomposes [nonuniformly distributed light power] for both diffuse and sh to a [uniform light power] * [visiblity map]
        # add to more output channel, vis_diffuse and vis_sh
        # Log: *_visibility
        # Results: it seems the result is better than rgb = lt_diffuse_pw * albedo * lt_diffuse + lt_sh_pw * lt_sh
        # Commit: 3f0a91cfa8c3fcc6b8ac498178b59ef58420de4d
            # lt_diffuse_lit = lt_diffuse_pw * albedo * lt_diffuse
            # lt_sh_lit = lt_sh_pw * albedo * lt_sh
            # lt_spec = spec * lt_diffuse
            # rgb = lt_vis_diffuse * lt_diffuse_lit + lt_vis_sh * lt_sh_lit
            # loss = 2*loss_img + loss_albedo + loss_normal + loss_albedo0 + loss_normal0

        # 2021/01/23
        # change near and far from 35 85 to 20 100 and change N_importance from 64 to 128
        # Log: *_near20_far100_Nimt128
        # Results: compared with experiments on 2021/01/22, no improvement

        # 2021/01/24
        # update rendering equation, and also change near far N_import back to 35 85 64, which will faster training
        # Log: *_spec
        # Commit: 8dff0baa64953442eda1188222f88689cdfce25e
        # Results: the results is better than previous experiments expecially on the forehead
        lt_diffuse_lit = lt_diffuse_pw * lt_diffuse
        lt_sh_lit = lt_sh_pw * lt_sh
        lt_spec = spec * lt_diffuse
        rgb = (lt_vis_diffuse * lt_diffuse_lit + lt_sh_lit) * albedo + lt_spec
        # rgb = tf.math.sigmoid(raw[..., :3])

        # 2021/01/24
        # output specular map as well as visbility map and leave the equation unchanged
        # Log: *_spec
        # Commit: 17bac7cd90defd27482631402dc62b4b1797704c
        # Results: work not good

        # 2021/01/26
        # use raw_material net to predict albedo and normal, and try to predict a clean rgb
        # Log: *_material_net
        # Commit: 16aa9dee580e28fb67c3835435e206cc0d3f07a6
        # Results: albedo and normal cannot be predict precisely by the material net

        # Compute weight for RGB of each sample along each ray.  A cumprod() is
        # used to express the idea of the ray not having reflected up to this
        # sample yet.
        # [N_rays, N_samples]
        weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)
        # weights_material = alpha_material * tf.math.cumprod(1.-alpha_material + 1e-10, axis=-1, exclusive=True)

        # Computed weighted color of each sample along each ray.
        rgb_map = tf.reduce_sum(
            weights[..., None] * rgb, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]
        albedo_map = tf.reduce_sum(
            weights[..., None] * albedo, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]
        norm_map = tf.reduce_sum(
            weights[..., None] * norm, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]
        diffuse_map = tf.reduce_sum(
            weights[..., None] * lt_diffuse, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]
        sh_map = tf.reduce_sum(
            weights[..., None] * lt_sh, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]
        diffuse_lit_map = tf.reduce_sum(
            weights[..., None] * lt_diffuse_lit, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]
        sh_lit_map = tf.reduce_sum(
            weights[..., None] * lt_sh_lit, axis=-2) * mask[..., None] + (1.-mask[..., None])  # [N_rays, 3]

        # Estimated depth map is expected distance.
        depth_map = tf.reduce_sum(weights * z_vals, axis=-1) * mask + (1.-mask)
        spec_map = tf.reduce_sum(weights * np.squeeze(spec), axis=-1) * mask + (1.-mask)
        diffuse_vis_map = tf.reduce_sum(weights * np.squeeze(lt_vis_diffuse), axis=-1) * mask + (1.-mask)

        # Disparity map is inverse depth.
        # disp_map = 1./tf.maximum(1e-10, depth_map /
        #                          tf.reduce_sum(weights, axis=-1))

        # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
        acc_map = tf.reduce_sum(weights, -1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            # rgb_map = rgb_map + (1.-acc_map[..., None])
            # albedo_map = albedo_map + (1.-acc_map[..., None])
            # diffuse_map = diffuse_map + (1.-acc_map[..., None])
            # norm_map = norm_map + (1.-acc_map[..., None])
            # sh_map = sh_map + (1.-acc_map[..., None])
            pass

        return rgb_map, albedo_map, norm_map, diffuse_map, sh_map, diffuse_lit_map, sh_lit_map, depth_map, spec_map, diffuse_vis_map, weights

    # def raws2outputs(raws, z_vals, rays_d, sh, light_probe):
    #     """Transforms model's predictions to semantically meaningful values.

    #     Args:
    #       raw: [num_rays, num_samples along ray, 4]. Prediction from model.
    #       z_vals: [num_rays, num_samples along ray]. Integration time.
    #       rays_d: [num_rays, 3]. Direction of each ray.

    #     Returns:
    #       rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
    #       disp_map: [num_rays]. Disparity map. Inverse of depth map.
    #       acc_map: [num_rays]. Sum of weights along each ray.
    #       weights: [num_rays, num_samples]. Weights assigned to each sampled color.
    #       depth_map: [num_rays]. Estimated distance to object.
    #     """
    #     # Function for computing density from model prediction. This value is
    #     # strictly between [0, 1].
    #     def raw2alpha(raw, dists, act_fn=tf.nn.relu): return 1.0 - \
    #         tf.exp(-act_fn(raw) * dists)

    #     raw = raws['raw']
    #     raw_posx = raws['raw_posx']
    #     raw_posy = raws['raw_posy']
    #     raw_posz = raws['raw_posz']
    #     # raw_negx = raws['raw_negx']
    #     # raw_negy = raws['raw_negy']
    #     # raw_negz = raws['raw_negz']
    #     # raw_material = raws['raw_material']

    #     # Compute 'distance' (in time) between each integration time along a ray.
    #     dists = z_vals[..., 1:] - z_vals[..., :-1]

    #     # The 'distance' from the last integration time is infinity.
    #     dists = tf.concat(
    #         [dists, tf.broadcast_to([1e10], dists[..., :1].shape)],
    #         axis=-1)  # [N_rays, N_samples]

    #     # Multiply each distance by the norm of its corresponding direction ray
    #     # to convert to real world distance (accounts for non-unit directions).
    #     dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

    #     # Extract albedo of each sample position along each ray.
    #     rgb = tf.math.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

    #     # Add noise to model's predictions for density. Can be used to
    #     # regularize network during training (prevents floater artifacts).
    #     noise = 0.
    #     if raw_noise_std > 0.:
    #         noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

    #     # Predict density of each sample along each ray. Higher values imply
    #     # higher likelihood of being absorbed at this point.
    #     alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    #     # alpha_posx = raw2alpha(raw_posx[..., 3] + noise, dists)
    #     # alpha_posy = raw2alpha(raw_posy[..., 3] + noise, dists)
    #     # alpha_posz = raw2alpha(raw_posz[..., 3] + noise, dists)
    #     # alpha_negx = raw2alpha(raw_negx[..., 3] + noise, dists)
    #     # alpha_negy = raw2alpha(raw_negy[..., 3] + noise, dists)
    #     # alpha_negz = raw2alpha(raw_negz[..., 3] + noise, dists)

    #     # Compute weight for RGB of each sample along each ray.  A cumprod() is
    #     # used to express the idea of the ray not having reflected up to this
    #     # sample yet.
    #     # [N_rays, N_samples]
    #     weights = alpha * \
    #         tf.math.cumprod(1.-alpha + 1e-10, axis=-1, exclusive=True)
    #     # weights_posx = alpha_posx * tf.math.cumprod(1.-alpha_posx + 1e-10, axis=-1, exclusive=True)
    #     # weights_posy = alpha_posy * tf.math.cumprod(1.-alpha_posy + 1e-10, axis=-1, exclusive=True)
    #     # weights_posz = alpha_posz * tf.math.cumprod(1.-alpha_posz + 1e-10, axis=-1, exclusive=True)
    #     # weights_negx = alpha_negx * tf.math.cumprod(1.-alpha_negx + 1e-10, axis=-1, exclusive=True)
    #     # weights_negy = alpha_negy * tf.math.cumprod(1.-alpha_negy + 1e-10, axis=-1, exclusive=True)
    #     # weights_negz = alpha_negz * tf.math.cumprod(1.-alpha_negz + 1e-10, axis=-1, exclusive=True)

    #     # Estimated depth map is expected distance.
    #     depth_map = tf.reduce_sum(weights * z_vals, axis=-1) # [N_rays,]
    #     # depth_map_posx = tf.reduce_sum(weights_posx * z_vals, axis=-1)
    #     # depth_map_posy = tf.reduce_sum(weights_posy * z_vals, axis=-1)

    #     # https://stackoverflow.com/questions/53350391/surface-normal-calculation-from-depth-map-in-python normal computation
    #     # https://answers.opencv.org/question/82453/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-product/
    #     # zy, zx = np.gradient(depth_map)
    #     # zx = np.gradient(depth_map, depth_map_posx)
    #     # zy = np.gradient(depth_map, depth_map_posy)
    #     # norm = np.dstack((-zx, -zy, np.ones_like(depth_map)))
    #     # norm = norm / (tf.norm(norm, axis=2, keepdims=True) + 1e-6) # [N_rays, N_samples, 3]

    #     # compute normal
    #     dens = tf.maximum(raw[..., -1], 0.)
    #     dens_posx = tf.maximum(raw_posx[..., -1], 0.)
    #     dens_posy = tf.maximum(raw_posy[..., -1], 0.)
    #     dens_posz = tf.maximum(raw_posz[..., -1], 0.)
    #     # dens_negx = tf.maximum(raw_negx[..., -1], 0.)
    #     # dens_negy = tf.maximum(raw_negy[..., -1], 0.)
    #     # dens_negz = tf.maximum(raw_negz[..., -1], 0.)

    #     norm_x = dens - dens_posx
    #     norm_y = dens - dens_posy
    #     norm_z = dens - dens_posz
    #     # norm_x = (dens - dens_posx) + (dens_negx - dens)
    #     # norm_y = (dens - dens_posy) + (dens_negy - dens)
    #     # norm_z = (dens - dens_posz) + (dens_negz - dens)

    #     norm = tf.stack([norm_x, norm_y, norm_z], axis=-1)
    #     norm = norm / (tf.norm(norm, axis=2, keepdims=True) + 1e-6)  # [N_rays, N_samples, 3]
    #     norm_x, norm_y, norm_z = tf.unstack(norm, axis=2)
    #     norm_x2, norm_y2, norm_z2 = norm_x*norm_x, norm_y*norm_y, norm_z*norm_z

    #     # Extract sphereical harmoncis coefficients
    #     sh_basis = [
    #         # level 0
    #         tf.cast(tf.broadcast_to(1.0 / 2.0 * np.sqrt(1.0 / np.pi), norm_x.shape.as_list()), tf.float32), # l = 0; m = 0
    #         # level 1
    #         np.sqrt(3.0 / (4.0 * np.pi)) * norm_y,  # l = 1; m = -1
    #         np.sqrt(3.0 / (4.0 * np.pi)) * norm_z,  # l = 1; m = 0
    #         np.sqrt(3.0 / (4.0 * np.pi)) * norm_x,  # l = 1; m = 1
    #         # level 2
    #         1.0 / 2.0 * np.sqrt(15.0 / np.pi) * norm_x * norm_y,
    #         1.0 / 2.0 * np.sqrt(15.0 / np.pi) * norm_z * norm_y,
    #         1.0 / 4.0 * np.sqrt(5.0 / np.pi)  * (-norm_x2-norm_y2 + 2.0*norm_z2),
    #         1.0 / 2.0 * np.sqrt(15.0 / np.pi) * norm_x * norm_z,
    #         1.0 / 4.0 * np.sqrt(15.0 / np.pi) * norm_x2 - norm_y2,
    #         # level 3
    #         1.0 / 4.0 * np.sqrt(35.0 / (2.0 * np.pi)) * (3.0 * norm_x2 - norm_y2) * norm_y,
    #         1.0 / 2.0 * np.sqrt(105.0 / np.pi)        * norm_x * norm_z * norm_y,
    #         1.0 / 4.0 * np.sqrt(21.0 / (2.0 * np.pi)) * norm_y * (5.0*norm_z2 - norm_x2 - norm_y2),
    #         1.0 / 4.0 * np.sqrt(7.0 / np.pi)          * norm_z * (1.5*norm_z2 - 3.0*norm_x2 - 3.0*norm_y2),
    #         1.0 / 4.0 * np.sqrt(21.0 / (2.0 * np.pi)) * norm_x * (5.0*norm_z2 - norm_x2 - norm_y2),
    #         1.0 / 4.0 * np.sqrt(105.0 / np.pi)        * (norm_x2 - norm_y2) * norm_z,
    #         1.0 / 4.0 * np.sqrt(35.0 / (2.0 * np.pi)) * (norm_x2 - 3.0*norm_y2) * norm_x
    #     ]
    #     sh_basis = tf.stack(sh_basis, axis=-1) # [N_rays, N_samples, 16]
    #     sh = tf.broadcast_to(sh, sh_basis.shape.as_list()[:2] + list(sh.shape)) # [N_rays, N_samples, 16, 3]
    #     sh_light = tf.reduce_sum(sh * sh_basis[..., None], axis=-2) # [N_rays, N_samples, 3]

    #     # for direct light
    #     # down_step = 100
    #     # light_probe = light_probe[::down_step,::down_step,:] # [h,w,3]

    #     # get uv coordinates
    #     h, w, _ = light_probe.shape
    #     u = np.arange(w)/w
    #     v = np.arange(h)/h

    #     # get spherical coordinates
    #     r = 1
    #     rot = 0.25
    #     theta = v * np.pi # 0 to pi
    #     phi = (u+rot) * 2*np.pi # -pi to pi

    #     # spherical coordinates to cartesian coordinates
    #     X = r * np.sin(theta[..., None]) * np.cos(phi) # [h,w]
    #     Y = r * np.sin(theta[..., None]) * np.sin(phi) # [h,w]
    #     Z = r * np.cos(theta[..., None]) * np.ones(phi.shape) # [h,w]
    #     x = np.reshape(X, -1)
    #     y = np.reshape(Y, -1)
    #     z = np.reshape(Z, -1)

    #     # map uv to pixel scale
    #     # m = np.ceil(u*w)
    #     # n = np.ceil(v*h)

    #     # get color from light probe using 
    #     l_power = 30.0
    #     sh_power = 1.0
    #     l_dir = np.stack([x, y, z], axis=-1).astype(np.float32) # [h*w,3]
    #     l_weight = np.sin(theta) # [h,]
    #     l_color = np.reshape(light_probe * l_weight[:, None, None], (-1,3)).astype(np.float32) # [h*w,3]
    #     nDotL = tf.maximum(tf.matmul(norm, l_dir, transpose_b=True) / l_color.shape[0], 0.) # [N_rays, N_samples, 3] * [3, h*w] -> [N_rays, N_samples, h*w]
    #     lt_diffuse = tf.matmul(nDotL, l_color) # [N_rays, N_samples, h*w] * [h*w,3] -> [N_rays, N_samples, 3]

    #     # rgb = albedo * lt_diffuse + sh_light
    #     albedo = (rgb - sh_power * sh_light) / (l_power * lt_diffuse + 1e-5)

    #     # Computed weighted color of each sample along each ray.
    #     rgb_map = tf.reduce_sum(
    #         weights[..., None] * rgb, axis=-2)  # [N_rays, 3]
    #     albedo_map = tf.reduce_sum(
    #         weights[..., None] * albedo, axis=-2)  # [N_rays, 3]
    #     diffuse_map = tf.reduce_sum(
    #         weights[..., None] * lt_diffuse, axis=-2) * 5.0  # [N_rays, 3]
    #     norm_map = tf.reduce_sum(
    #         weights[..., None] * norm, axis=-2)  # [N_rays, 3]
    #     sh_map = tf.reduce_sum(
    #         weights[..., None] * sh_light, axis=-2)  # [N_rays, 3]

    #     # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    #     acc_map = tf.reduce_sum(weights, -1)

    #     # To composite onto a white background, use the accumulated alpha map.
    #     if white_bkgd:
    #         rgb_map = rgb_map + (1.-acc_map[..., None])

    #     # return rgb_map, albedo_map, sh_map, spec_map, sh_coef_out, disp_map, acc_map, weights, depth_map
    #     return rgb_map, albedo_map, diffuse_map, norm_map, sh_map, depth_map, weights

    ###############################
    # batch size
    N_rays = ray_batch.shape[0]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = tf.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    mask = ray_batch[:, 8]
    albedo_gt = ray_batch[:, 9:12]
    normal_gt = ray_batch[:, 12:15]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = tf.linspace(0., 1., N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = tf.broadcast_to(z_vals, [N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    # ray_w = 2 # ray width, the smaller the wider
    # delta = tf.reduce_mean(far-near).numpy() / (N_samples * ray_width)
    # pts_o = rays_o[..., None, :]
    # pts_d = rays_d[..., None, :]
    # pts_z = z_vals[..., :, None]
    # offset_x = tf.cast(tf.broadcast_to([delta, 0, 0], pts_o.shape.as_list()), tf.float32)
    # offset_y = tf.cast(tf.broadcast_to([0, delta, 0], pts_o.shape.as_list()), tf.float32)
    # offset_z = tf.cast(tf.broadcast_to([0, 0, delta], pts_o.shape.as_list()), tf.float32)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    # pts = pts_o + pts_d * z_vals[..., :, None]  # [N_rays, N_samples, 3]
    # pts_posx = pts_o + offset_x + pts_d * pts_z  # [N_rays, N_samples, 3]
    # pts_posy = pts_o + offset_y + pts_d * pts_z  # [N_rays, N_samples, 3]
    # pts_posz = pts_o + offset_z + pts_d * pts_z  # [N_rays, N_samples, 3]
    # pts_negx = pts_o - offset_x + pts_d * pts_z  # [N_rays, N_samples, 3]
    # pts_negy = pts_o - offset_y + pts_d * pts_z  # [N_rays, N_samples, 3]
    # pts_negz = pts_o - offset_z + pts_d * pts_z  # [N_rays, N_samples, 3]

    # Evaluate model at each point.
    # [N_rays, N_samples, 4] -> [N_rays, N_samples, 8]
    # raw = network_query_fn(pts, viewdirs, sh, network_fn)
    raw = network_query_fn(pts, viewdirs, network_fn)
    # raw_norm = network_query_fn(pts, viewdirs, network_material)
    # raws = {}
    # raws['raw'] = raw
    # raws['raw_posx'] = network_query_fn(pts_posx, viewdirs, network_fn)
    # raws['raw_posy'] = network_query_fn(pts_posy, viewdirs, network_fn)
    # raws['raw_posz'] = network_query_fn(pts_posz, viewdirs, network_fn)
    # raws['raw_negx'] = network_query_fn(pts_negx, viewdirs, network_fn)
    # raws['raw_negy'] = network_query_fn(pts_negy, viewdirs, network_fn)
    # raws['raw_negz'] = network_query_fn(pts_negz, viewdirs, network_fn)
    # if network_fn_ is not None:
        # raw_ = network_query_fn_(pts, viewdirs, sh, network_fn_)
        # raw_ = network_query_fn_(pts, viewdirs, network_fn_)
        # rgb_map, albedo_map, sh_map, spec_map, sh_coef_out, disp_map, acc_map, weights, depth_map = raw2outputs(
        #     raw, z_vals, rays_d, raw_)
        # rgb_map, albedo_map, weights = raw2outputs(raw, z_vals, rays_d, raw_)
    # else:
        # rgb_map, albedo_map, sh_map, spec_map, sh_coef_out, disp_map, acc_map, weights, depth_map = raw2outputs(
        #     raw, z_vals, rays_d)
        # rgb_map, albedo_map, diffuse_map, norm_map, sh_map, weights = raws2outputs(
        #     raws, z_vals, rays_d, sh, light_probe)
    rgb_0, _, weights = raw2outputs(raw, z_vals, rays_d, mask)
    # albedo_0, normal_0, weights = raw2outputs(raw, z_vals, rays_d)
    # _, normal_0, weights = raw2outputs(raw_norm, z_vals, rays_d)

    if N_importance > 0:

        # rgb_0, albedo_0, diffuse_0, norm_0, sh_light_0 = rgb_map, albedo_map, diffuse_map, norm_map, sh_map
        # rgb_0 = rgb_map

        # Obtain additional integration times to evaluate based on the weights
        # assigned to colors in the coarse model.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.))
        z_samples = tf.stop_gradient(z_samples)

        # Obtain all points to evaluate color, density at.
        z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)

        delta = tf.reduce_mean(far-near).numpy() / ((N_samples + N_importance) * ray_width)
        pts_o = rays_o[..., None, :]
        pts_d = rays_d[..., None, :]
        pts_z = z_vals[..., :, None]
        # offset_x = tf.cast(tf.broadcast_to([delta, 0, 0], pts_o.shape.as_list()), tf.float32)
        # offset_y = tf.cast(tf.broadcast_to([0, delta, 0], pts_o.shape.as_list()), tf.float32)
        # offset_z = tf.cast(tf.broadcast_to([0, 0, delta], pts_o.shape.as_list()), tf.float32)
        pts = pts_o + pts_d * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
        # pts_posx = pts_o + offset_x + pts_d * pts_z  # [N_rays, N_samples + N_importance, 3]
        # pts_posy = pts_o + offset_y + pts_d * pts_z  # [N_rays, N_samples + N_importance, 3]
        # pts_posz = pts_o + offset_z + pts_d * pts_z  # [N_rays, N_samples + N_importance, 3]
        # pts_negx = pts_o - offset_x + pts_d * pts_z  # [N_rays, N_samples + N_importance, 3]
        # pts_negy = pts_o - offset_y + pts_d * pts_z  # [N_rays, N_samples + N_importance, 3]
        # pts_negz = pts_o - offset_z + pts_d * pts_z  # [N_rays, N_samples + N_importance, 3]

        # Make predictions with network_fine.
        run_occupancy = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, run_occupancy)
        raw_material = raw
        # raw_material = network_query_fn(pts, viewdirs, network_material)
        # raws = {}
        # raws['raw'] = raw
        # raws['raw_posx'] = network_query_fn(pts_posx, viewdirs, run_occupancy)
        # raws['raw_posy'] = network_query_fn(pts_posy, viewdirs, run_occupancy)
        # raws['raw_posz'] = network_query_fn(pts_posz, viewdirs, run_occupancy)
        # raws['raw_negx'] = network_query_fn(pts_negx, viewdirs, run_occupancy)
        # raws['raw_negy'] = network_query_fn(pts_negy, viewdirs, run_occupancy)
        # raws['raw_negz'] = network_query_fn(pts_negz, viewdirs, run_occupancy)
        # raws['raw_material'] = network_query_fn(pts, viewdirs, run_material)
        if network_fn_ is not None:
            # run_fn_ = network_fn_ if network_fine_ is None else network_fine_
            # raw_ = network_query_fn_(pts, viewdirs, run_fn_)
            # rgb_map, albedo_map, weights = raw2outputs(
            #     raw, z_vals, rays_d, raw_)
            pass
        else:
            # rgb_1, _ = raw2outputs(raw, z_vals, rays_d)
            # rgb_map, albedo_map, diffuse_map, norm_map, sh_map, depth_map, weights = raws2outputs(
            #     raws, z_vals, rays_d, sh, light_probe)
            rgb_map, albedo_map, norm_map, diffuse_map, sh_map, diffuse_lit_map, sh_lit_map, depth_map, spec_map, diffuse_vis_map, weights = raw2outputs_test(
                raw, raw_material, z_vals, rays_d, sh, light_probe, mask, albedo_gt, normal_gt)

    # ret = {'rgb_map': rgb_map, 'albedo_map': albedo_map, 'sh_map': sh_map, 'spec_map': spec_map, 'sh_coef_out': sh_coef_out,
    #        'disp_map': disp_map, 'acc_map': acc_map}
    ret = {'rgb_map': rgb_map, 'albedo_map': albedo_map, 'norm_map': norm_map, 'diffuse_map': diffuse_map,
           'sh_map': sh_map, 'diffuse_lit_map': diffuse_lit_map, 'sh_lit_map': sh_lit_map, 'depth_map': depth_map, 'spec_map': spec_map, 'diffuse_vis_map': diffuse_vis_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_0
        # ret['albedo0'] = albedo_0
        # ret['normal0'] = normal_0
        # ret['rgb1'] = rgb_1
        # ret['albedo0'] = albedo_0
        # ret['diffuse0'] = diffuse_0
        # ret['norm0'] = norm_0
        # ret['sh_light_0'] = sh_light_0
        # ret['spec0'] = spec_0
        # ret['sh_coef_0'] = sh_coef_0
        # ret['disp0'] = disp_map_0
        # ret['acc0'] = acc_map_0
        ret['z_std'] = tf.math.reduce_std(z_samples, -1)  # [N_rays]

    for k in ret:
        tf.debugging.check_numerics(ret[k], 'output {}'.format(k))

    return ret


def batchify_rays(rays_flat, sh, light_probe, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM."""
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], sh, light_probe, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: tf.concat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, focal,
           chunk=1024*32, rays=None, c2w=None, ndc=True,
           sh=None,
           light_probe=None,
           mask=None,
           albedo_gt=None,
           normal_gt=None,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays

    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.

    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)

        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)
        viewdirs = tf.cast(tf.reshape(viewdirs, [-1, 3]), dtype=tf.float32)

    if sh is not None:
        pass

    shp = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(
            H, W, focal, tf.cast(1., tf.float32), rays_o, rays_d)

    # Create ray batch
    rays_o = tf.cast(tf.reshape(rays_o, [-1, 3]), dtype=tf.float32)
    rays_d = tf.cast(tf.reshape(rays_d, [-1, 3]), dtype=tf.float32)
    near, far = near * \
        tf.ones_like(rays_d[..., :1]), far * tf.ones_like(rays_d[..., :1])
    rays_m = tf.cast(tf.reshape(mask, [-1, 1]), dtype=tf.float32)
    rays_a = tf.cast(tf.reshape(albedo_gt, [-1, 3]), dtype=tf.float32)
    rays_n = tf.cast(tf.reshape(normal_gt, [-1, 3]), dtype=tf.float32)

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = tf.concat([rays_o, rays_d, near, far, rays_m, rays_a, rays_n], axis=-1)
    if use_viewdirs:
        # (ray origin, ray direction, min dist, max dist, normalized viewing direction)
        rays = tf.concat([rays, viewdirs], axis=-1)

    # Render and reshape
    all_ret = batchify_rays(rays, sh, light_probe, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(shp[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = tf.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'albedo_map', 'norm_map', 'diffuse_map', 'sh_map', 'diffuse_lit_map', 'sh_lit_map', 'depth_map', 'spec_map', 'diffuse_vis_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwfs, shs, light_probes, masks, albedos_gt, normals_gt, chunk, render_kwargs, names=None, gt_imgs=None, savedir=None, render_factor=0):

    # H, W, focal = hwf

    # if render_factor != 0:
    #     # Render downsampled for speed
    #     H = H//render_factor
    #     W = W//render_factor
    #     focal = focal/render_factor

    # if shs.ndim == 2:
    #     shs = shs[np.newaxis, ...]
    # if hwfs.ndim == 1:
    #     hwfs = hwfs[np.newaxis, ...]

    rgbs = []
    albedos = []
    norms = []
    diffuses = []
    sh_lights = []
    diffuse_lits = []
    sh_lits = []
    depths = []
    specs = []
    diffuse_vises = []

    t = time.time()
    for i, c2w in enumerate(render_poses):
        # timer
        print(i, time.time() - t)
        t = time.time()

        # prepare spherical harmonics
        if shs.ndim == 2:
            sh = shs
            light_probe = light_probes
            mask = masks
            albedo_gt = albedos_gt
            normal_gt = normals_gt
        else:
            # sh = shs[i, :4, :3]
            sh = shs[i, :16, :3]
            light_probe = light_probes[i]
            mask = masks[i]
            albedo_gt = albedos_gt[i]
            normal_gt = normals_gt[i]

        # prepare hwf
        if hwfs.ndim == 1:
            H, W, focal = hwfs
        else:
            H, W, focal = hwfs[i]
        H, W = int(H), int(W)
        if render_factor != 0:
            # Render downsampled for speed
            H = H//render_factor
            W = W//render_factor
            focal = focal/render_factor

        # render
        rgb, albedo, norm, diffuse, sh_light, diffuse_lit, sh_lit, depth, spec, diffuse_vis, _ = render(
            H, W, focal, chunk=chunk, c2w=c2w[:3, :4], sh=sh, light_probe=light_probe, mask=mask, albedo_gt=albedo_gt, normal_gt=normal_gt, **render_kwargs)

        #
        rgbs.append(rgb.numpy())
        albedos.append(albedo.numpy())
        norms.append(norm.numpy())
        diffuses.append(diffuse.numpy())
        sh_lights.append(sh_light.numpy())
        diffuse_lits.append(diffuse_lit.numpy())
        sh_lits.append(sh_lit.numpy())
        depths.append(depth.numpy())
        specs.append(spec.numpy())
        diffuse_vises.append(diffuse_vis.numpy())
        if i == 0:
            print(rgb.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            print(p)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            albedo8 = to8b(albedos[-1])
            norm8 = to8b(norms[-1])
            diffuse8= to8b(diffuses[-1])
            sh_light8 = to8b(sh_lights[-1])
            diffuse_lit8 = to8b(diffuse_lits[-1])
            sh_lit8 = to8b(sh_lits[-1])
            depth8 = to8b(depths[-1])
            spec8 = to8b(specs[-1])
            diffuse_vis8 = to8b(diffuse_vises[-1])
            # filename = os.path.join(savedir, '{:03d}_{}.png'.format(i, names[i]))
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}.png'.format(i, names[i])), rgb8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_albedo.png'.format(i, names[i])), albedo8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_norm.png'.format(i, names[i])), norm8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_diffuse.png'.format(i, names[i])), diffuse8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_sh_light.png'.format(i, names[i])), sh_light8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_diffuse_lit.png'.format(i, names[i])), diffuse_lit8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_sh_lit.png'.format(i, names[i])), sh_lit8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_depth.png'.format(i, names[i])), depth8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_spec.png'.format(i, names[i])), spec8)
            imageio.imwrite(os.path.join(
                savedir, '{:03d}_{}_diffuse_vis.png'.format(i, names[i])), diffuse_vis8)

    rgbs = np.stack(rgbs, 0)
    albedos = np.stack(albedos, 0)
    diffuses = np.stack(diffuses, 0)
    norms = np.stack(norms, 0)
    sh_lights = np.stack(sh_lights, 0)
    depths = np.stack(depths, 0)
    specs = np.stack(specs, 0)
    diffuse_vises = np.stack(diffuse_vises, 0)

    return rgbs, albedos, norms, diffuses, sh_lights, diffuse_lits, sh_lits, depths, specs, diffuse_vises


def create_nerf(args):
    """Instantiate NeRF's MLP model."""

    # positional encoding for xyz
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    # positional encoding for view
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed)

    # for sh, no need for positional encoding
    # input_ch_sh = 12
    input_ch_sh = 48

    output_ch = 4  # r, g, b, sigma
    # output_ch = 16  # r, g, b, sigma, rs00, rs10, rs11, rs12, gs00, gs10, gs11, gs12, bs00, bs10, bs11, bs12
    # output_ch = 17  # r, g, b, sigma, rs00, rs10, rs11, rs12, gs00, gs10, gs11, gs12, bs00, bs10, bs11, bs12, specular
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, input_ch_sh=input_ch_sh)
    # grad_vars_material = model.trainable_variables
    grad_vars = model.trainable_variables
    models = {'model': model}

    model_fine = None
    model_material = None
    if args.N_importance > 0:
        model_fine = init_nerf_model(
            D=args.netdepth_fine, W=args.netwidth_fine,
            input_ch=input_ch, output_ch=output_ch, skips=skips,
            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, input_ch_sh=input_ch_sh)
        grad_vars += model_fine.trainable_variables
        models['model_fine'] = model_fine

        # model_material = init_nerf_model(
        #     D=args.netdepth_fine, W=args.netwidth_fine,
        #     input_ch=input_ch, output_ch=output_ch, skips=skips,
        #     input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, input_ch_sh=input_ch_sh)
        # grad_vars_material += model_material.trainable_variables
        # models['model_material'] = model_material

    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        # 'network_material': model_material,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
        'ray_width': args.ray_width,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        model.set_weights(np.load(ft_weights, allow_pickle=True))
        start = int(ft_weights[-10:-4]) + 1
        print('Resetting step to', start)

        if model_fine is not None:
            ft_weights_fine = '{}_fine_{}'.format(ft_weights[:-11], ft_weights[-10:])
            print('Reloading fine from', ft_weights_fine)
            model_fine.set_weights(np.load(ft_weights_fine, allow_pickle=True))

        # if model_material is not None:
        #     ft_weights_material = '{}_material_{}'.format(ft_weights[:-11], ft_weights[-10:])
        #     print('Reloading material from', ft_weights_material)
        #     model_material.set_weights(np.load(ft_weights_material, allow_pickle=True))

    # return render_kwargs_train, render_kwargs_test, start, grad_vars, grad_vars_material, models
    return render_kwargs_train, render_kwargs_test, start, grad_vars, models


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str,
                        default='./data/llff/fern', help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int,
                        default=8, help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')

    # pre-crop options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--ray_width", type=float, default=3.,
                        help='distance between rays when performing ray marching from the same direction')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape,
              render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = tf.reduce_min(bds) * .9
            far = tf.reduce_max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        # images, poses, render_poses, hwf, i_split = load_blender_data(
        #     args.datadir, args.half_res, args.testskip)
        # print('Loaded blender', images.shape,
        #       render_poses.shape, hwf, args.datadir)
        
        images, names, poses, hwfs, shs, render_poses, i_split = load_blender_data_fill_up(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, poses.shape,
            hwfs.shape, shs.shape, render_poses.shape, args.datadir)

        i_train, i_val, i_test = i_split
        
        light_probe = imageio.imread('/glab2/Users/jyang/data/nerf_synthesic/model_0_sh_21/light/equirectangular.exr')

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape,
              render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'lightstage':

        # load data
        images, light_probes, albedos_gt, normals_gt, names, poses, hwfs, shs, render_poses, i_split = load_lightstage_data(
            basedir=args.datadir, half_res=args.half_res, testskip=args.testskip)
        print('Loaded lightstage', images.shape, poses.shape,
              hwfs.shape, shs.shape, render_poses.shape, args.datadir)

        # split traning, test and validation
        i_train, i_val, i_test = i_split

        # set near and far value, real distance
        # model 0 near 100 far 400
        near = 35.
        far = 85.

        # load light probe here for now
        # light_probe = imageio.imread(args.datadir+'/light/equirectangular.exr')
        # gain = 1.0
        # gamma = 1.5
        # light_probe = gain * (light_probe ** gamma) # gamma correction, gamma 2.2 is youtube default gamma
        # light_probe = np.minimum(light_probe, 10.0) # filter way brighter light samples
        
        # light_probe = imageio.imread('/glab2/data/Users/jyang/data/nerf_synthesic/model_0_sh_21/light/equirectangular.exr')

        # set white_bkgd if alpha channel is available
        if args.white_bkgd:
            masks = images[..., -1:]
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
            albedos_gt = albedos_gt[..., :3]*albedos_gt[..., -1:] + (1.-albedos_gt[..., -1:])
            normals_gt = normals_gt[..., :3]*normals_gt[..., -1:] + (1.-normals_gt[..., -1:])
            # images = images[..., :3]*images[..., -1:]
            # albedos_gt = albedos_gt[..., :3]*albedos_gt[..., -1:]
            # normals_gt = normals_gt[..., :3]*normals_gt[..., -1:]
        else:
            images = images[..., :3]

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    # H, W, focal = hwf
    # H, W = int(H), int(W)
    # hwf = [H, W, focal]
    hwf_avg = np.average(hwfs, axis=0)
    sh_default = np.array([
        [0.638712, 0.284742, 0.118981],
        [-0.121597, -0.0644393, -0.0280195],
        [-0.0704244, -0.0440014, -0.0278942],
        [-0.00941534, -0.0196714, -0.0270952],
        [0.0244973, 0.019399, 0.0146708],
        [-0.0738909, -0.0249716, -0.00191948],
        [-0.112882, -0.0680578, -0.0321083],
        [-0.0018545, -0.00141121, -0.00426523],
        [0.121973, 0.109539, 0.0996329],
        [0.195734, 0.0611543, -0.0115104],
        [-0.0243289, -0.0109742, -0.00370652],
        [0.119068, 0.0567287, 0.0189506],
        [-0.0188627, -0.00730853, -0.0120698],
        [-0.0146965, -0.00352992, 0.00319044],
        [-0.00446196, -0.00805026, -0.00354193],
        [-0.000833787, -0.0136447, -0.0227599]
    ]).astype(np.float32)  # [16, 3] 03-Ueno-Shrine_3k.exr

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    # render_kwargs_train, render_kwargs_test, start, grad_vars, grad_vars_material, models = create_nerf(
    #     args)
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        args)

    bds_dict = {
        'near': tf.cast(near, tf.float32),
        'far': tf.cast(far, tf.float32),
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        if args.render_test:
            # render_test switches to test poses
            images = images[i_test]
        else:
            # Default is smoother render_poses path
            images = None

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.render_test else 'path', start))
        os.makedirs(testsavedir, exist_ok=True)
        print('test poses shape', render_poses.shape)

        # rgbs, _ = render_path(render_poses, hwf_avg, sh_default, args.chunk, render_kwargs_test,
        #                       gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
        print('Done rendering', testsavedir)
        # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'),to8b(rgbs), fps=30, quality=8)

        return

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    # optimizer_material = tf.keras.optimizers.Adam(lrate)
    # models['optimizer_material'] = optimizer_material

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # For random ray batching.
        #
        # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
        # interpreted as,
        #   axis=0: ray origin in world space
        #   axis=1: ray direction in world space
        #   axis=2: observed RGB color of pixel
        print('get rays')
        # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
        # for each pixel in the image. This stack() adds a new dimension.
        rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
        rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
        rays_rgb = np.stack([rays_rgb[i]
                             for i in i_train], axis=0)  # train images only
        # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)
        print('done')
        i_batch = 0

    N_iters = 1000000
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # Summary writers
    writer = tf.contrib.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start, N_iters):
        time0 = time.time()

        # Sample random ray batch

        if use_batching:
            # Random over all images
            batch = rays_rgb[i_batch:i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                np.random.shuffle(rays_rgb)
                i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(i_train)

            # get traning data
            target = images[img_i]
            target_albedo = albedos_gt[img_i]
            target_normal = normals_gt[img_i]
            target_mask = masks[img_i]
            pose = poses[img_i, :3, :4]
            # sh = shs[img_i, :4, :3]
            sh = shs[img_i, :16, :3]
            light_probe = light_probes[img_i]
            H, W, focal = hwfs[img_i]
            H, W = int(H), int(W)

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, pose)
                if i < args.precrop_iters:
                    dH = int(H//2 * args.precrop_frac)
                    dW = int(W//2 * args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH),
                        tf.range(W//2 - dW, W//2 + dW),
                        indexing='ij'), -1)
                    if i < 10:
                        print('precrop', dH, dW, coords[0, 0], coords[-1, -1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)
                target_a = tf.gather_nd(target_albedo, select_inds)
                target_n = tf.gather_nd(target_normal, select_inds)
                target_m = tf.gather_nd(target_mask, select_inds)

        #####  Core optimization loop  #####

        # with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as tape:
        # Make predictions for color, disparity, accumulated opacity.

            rgb, albedo, normal, _, _, _, _, _, _, _, extras = render(
                H, W, focal, chunk=args.chunk, rays=batch_rays, sh=sh, light_probe=light_probe, mask=target_m, albedo_gt=target_a, normal_gt=target_n,
                verbose=i < 10, retraw=True, **render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            loss_img = img2mse(rgb, target_s)
            loss_albedo = img2mse(albedo, target_a)
            loss_normal = img2mse(normal, target_n)
            # sh_loss = img2mse(sh_coef, sh_parm)
            trans = extras['raw'][..., -1]
            loss = loss_img + loss_normal
            # loss = loss_img + loss_albedo + loss_normal
            psnr = mse2psnr(loss)
            # psnr_img = mse2psnr(loss_img)
            # psnr_albedo = mse2psnr(loss_albedo)
            # psnr_normal = mse2psnr(loss_normal)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                loss_rgb0 = img2mse(extras['rgb0'], target_s)
                loss += loss_rgb0
            if 'albedo0' in extras:
                loss_albedo0 = img2mse(extras['albedo0'], target_a)
                loss += loss_albedo0
                # psnr_albedo0 = mse2psnr(loss_albedo0)
            if 'normal0' in extras:
                loss_normal0 = img2mse(extras['normal0'], target_n)
                loss += loss_normal0
                # psnr_normal0 = mse2psnr(loss_normal0)

        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        # gradients_material = tape.gradient(loss_material, grad_vars_material)
        # optimizer_material.apply_gradients(zip(gradients_material, grad_vars_material))

        dt = time.time()-time0

        #####           end            #####

        # Rest is logging

        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname, '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_video == 0 and i > 0:

            # rgbs, albedos, sh_lights, _ = render_path(
            #     render_poses, hwf_avg, sh_default, args.chunk, render_kwargs_test)
            # rgbs, _, _, _ = render_path(
            #     render_poses, hwf_avg, sh_default, args.chunk, render_kwargs_test)
            # print('Done, saving', rgbs.shape, disps.shape)
            # print('Done, saving', rgbs.shape, albedos.shape)
            moviebase = os.path.join(
                basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # imageio.mimwrite(moviebase + 'rgb.mp4',
            #                  to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4',
            #                  to8b(disps / np.max(disps)), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'albedo.mp4',
            #                  to8b(albedos), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'sh_light.mp4',
            #                  to8b(sh_lights), fps=30, quality=8)

            if args.use_viewdirs:
                render_kwargs_test['c2w_staticcam'] = render_poses[0][:3, :4]
                # rgbs_still, _, _, _ = render_path(
                #     render_poses, hwf_avg, sh_default, args.chunk, render_kwargs_test)
                # rgbs_still, _, _, _ = render_path(
                #     render_poses, hwf_avg, sh_default, args.chunk, render_kwargs_test)
                render_kwargs_test['c2w_staticcam'] = None
                # imageio.mimwrite(moviebase + 'rgb_still.mp4',
                #                  to8b(rgbs_still), fps=30, quality=8)

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            render_path(poses[i_test], hwfs[i_test], shs[i_test], light_probes[i_test], masks[i_test], albedos_gt[i_test], normals_gt[i_test], args.chunk, render_kwargs_test, names=names[i_test],
                        gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % args.i_print == 0 or i < 10:

            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))
            with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_print):
                tf.contrib.summary.scalar('loss', loss)
                tf.contrib.summary.scalar('psnr', psnr)
                tf.contrib.summary.histogram('tran', trans)
                if args.N_importance > 0:
                    tf.contrib.summary.scalar('loss_img', loss_img)
                    tf.contrib.summary.scalar('loss_albedo', loss_albedo)
                    tf.contrib.summary.scalar('loss_normal', loss_normal)
                    tf.contrib.summary.scalar('loss_rgb0', loss_rgb0)
                    # tf.contrib.summary.scalar('loss_albedo0', loss_albedo0)
                    # tf.contrib.summary.scalar('loss_normal0', loss_normal0)

            if i % args.i_img == 0:

                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                target = images[img_i]
                name = names[img_i]
                pose = poses[img_i, :3, :4]
                sh = shs[img_i, :16, :3]
                mask = masks[img_i]
                albedo_gt = albedos_gt[img_i]
                normal_gt = normals_gt[img_i]
                light_probe = light_probes[img_i]
                H, W, focal = hwfs[img_i]
                H, W = int(H), int(W)
                
                rgb, albedo, norm, diffuse, sh_light, diffuse_lit, sh_lit, depth, spec, diffuse_vis, extras = render(
                    H, W, focal, chunk=args.chunk, c2w=pose, sh=sh, light_probe=light_probe, mask=mask, albedo_gt=albedo_gt, normal_gt=normal_gt, **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, target))

                if i % 2*args.i_img == 0:
                # Save out the validation image for Tensorboard-free monitoring
                    testimgdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                    if i == 0:
                        os.makedirs(testimgdir, exist_ok=True)
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}.png'.format(i, name)), to8b(rgb))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_albedo.png'.format(i, name)), to8b(albedo))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_norm.png'.format(i, name)), to8b(norm))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_diffuse.png'.format(i, name)), to8b(diffuse))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_sh_light.png'.format(i, name)), to8b(sh_light))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_diffuse_lit.png'.format(i, name)), to8b(diffuse_lit))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_sh_lit.png'.format(i, name)), to8b(sh_lit))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_depth.png'.format(i, name)), to8b(depth[..., tf.newaxis]))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_spec.png'.format(i, name)), to8b(spec[..., tf.newaxis]))
                    imageio.imwrite(os.path.join(
                        testimgdir, '{:06d}_{}_diffuse_vis.png'.format(i, name)), to8b(diffuse_vis[..., tf.newaxis]))

                with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):

                    tf.contrib.summary.image('rgb', to8b(rgb)[tf.newaxis])
                    tf.contrib.summary.image('albedo', to8b(albedo)[tf.newaxis])
                    tf.contrib.summary.image('norm', to8b(norm)[tf.newaxis])
                    tf.contrib.summary.image('diffuse', to8b(diffuse)[tf.newaxis])
                    tf.contrib.summary.image('sh_light', to8b(sh_light)[tf.newaxis])
                    tf.contrib.summary.image('diffuse_lit', to8b(diffuse_lit)[tf.newaxis])
                    tf.contrib.summary.image('sh_lit', to8b(sh_lit)[tf.newaxis])
                    tf.contrib.summary.image('depth', depth[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image('spec', spec[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image('diffuse_vis', diffuse_vis[tf.newaxis, ..., tf.newaxis])
                    tf.contrib.summary.image('rgb_holdout', target[tf.newaxis])
                    # tf.contrib.summary.image(
                    #     'disp', disp[tf.newaxis, ..., tf.newaxis])
                    # tf.contrib.summary.image(
                    #     'acc', acc[tf.newaxis, ..., tf.newaxis])

                    tf.contrib.summary.scalar('psnr_holdout', psnr)

                if args.N_importance > 0:

                    with tf.contrib.summary.record_summaries_every_n_global_steps(args.i_img):
                        tf.contrib.summary.image('rgb0', to8b(extras['rgb0'])[tf.newaxis])
                        # tf.contrib.summary.image('albedo0', to8b(extras['albedo0'])[tf.newaxis])
                        # tf.contrib.summary.image('normal0', to8b(extras['normal0'])[tf.newaxis])
                        # tf.contrib.summary.image('rgb1', to8b(extras['rgb1'])[tf.newaxis])
                        # tf.contrib.summary.image('disp0', extras['disp0'][tf.newaxis, ..., tf.newaxis])
                        tf.contrib.summary.image('z_std', extras['z_std'][tf.newaxis, ..., tf.newaxis])

        global_step.assign_add(1)


if __name__ == '__main__':
    train()
