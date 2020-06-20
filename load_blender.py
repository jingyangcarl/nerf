import os
import tensorflow as tf
import numpy as np
import imageio 
import json




trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

trans_t_ = lambda t, h : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,h],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
    
def pose_spherical_(theta, phi, radius, height):
    c2w = trans_t_(radius, height)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


# def load_blender_data(basedir, half_res=False, testskip=1):
def load_blender_data(basedir, half_res=False, testskip=1, use_depth=False, white_bkgd=True):
    # splits = ['train', 'val', 'test']
    if use_depth:
        splits = ['train', 'val', 'test', 'depth_l_train', 'depth_l_val', 'depth_l_test']
    else:
        splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        # if s=='train' or testskip==0:
        if s=='train' or s=='depth_train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            # fname = os.path.join(basedir, frame['file_path'] + '.png')
            if 'depth' in s:
                fname = os.path.join(basedir, frame['file_path'] + '.exr')
                img = imageio.imread(fname)
                if white_bkgd:
                    img = img[..., :3] * img[..., -1:]
                else:
                    img = img[..., :3]
            else:
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                img = imageio.imread(fname)
                img = (img / 255.).astype(np.float32) # keep all 4 channels (RGBA)
                if white_bkgd:
                    img = img[..., :3] * img[..., -1:] + (1. - img[..., -1:])
                else:
                    img = img[..., :3]
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
        # imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        imgs = (np.array(imgs)).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    render_poses = tf.stack([pose_spherical_(angle, 0.0, 1.5, 1.1) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.
        
    return imgs, poses, render_poses, [H, W, focal], i_split


