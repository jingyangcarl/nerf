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

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # list initialization for all
    all_imgs = []
    all_poses = []
    all_hwf = []
    all_sh = []

    counts = [0]

    # for each split
    for s in splits:
        # get current split
        meta = metas[s]

        # list initialization for current split
        imgs = []
        poses = []
        hwf = []
        sh = []

        # set skip parameter
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        # read each frame
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            hwf.append(np.array(frame['hwf']))
            sh.append(np.array(frame['sh']))

        # type process
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        hwf = np.array(hwf).astype(np.float32)
        sh = np.array(sh).astype(np.float32)

        # prepare for serialization
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_hwf.append(hwf)
        all_sh.append(sh)
    
    # split index
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # serialization
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    hwf = np.concatenate(all_hwf, 0)
    sh = np.concatenate(all_sh, 0)
    
    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    # generate novel render poses
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        # H = H//2
        # W = W//2
        # focal = focal/2.
        hwf[:,0] = hwf[:,0] // 2 # height 
        hwf[:,1] = hwf[:,1] // 2 # width
        hwf[:,2] = hwf[:,2] / 2 # focal
        
    return imgs, poses, render_poses, hwf, i_split


