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

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.
        
    return imgs, poses, render_poses, [H, W, focal], i_split

def load_blender_data_fill_up(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # list initialization for all
    all_imgs = []
    all_names = []
    all_poses = []
    all_hwfs = []
    all_shs = []

    counts = [0]

    # for each split
    for s in splits:
        # get current split
        meta = metas[s]

        # list initialization for current split
        imgs = []
        names = []
        poses = []
        hwfs = []
        shs = []

        # set skip parameter
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        # read each frame
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            names.append(os.path.splitext(os.path.basename(fname))[0])
            poses.append(np.array(frame['transform_matrix']))

            H, W = imgs[0].shape[:2]
            camera_angle_x = float(meta['camera_angle_x'])
            focal = .5 * W / np.tan(.5 * camera_angle_x)
            hwf = [H, W, focal]
            sh = np.array([
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

            hwfs.append(hwf)
            shs.append(sh)

        # type process
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        names = np.array(names).astype(np.str)
        poses = np.array(poses).astype(np.float32)
        hwfs = np.array(hwfs).astype(np.float32)
        shs = np.array(shs).astype(np.float32)

        # prepare for serialization
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_names.append(names)
        all_poses.append(poses)
        all_hwfs.append(hwfs)
        all_shs.append(shs)
    
    # split index
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    # serialization
    imgs = np.concatenate(all_imgs, 0)
    names = np.concatenate(all_names, 0)
    poses = np.concatenate(all_poses, 0)
    hwfs = np.concatenate(all_hwfs, 0)
    shs = np.concatenate(all_shs, 0)
    
    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res:
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        # H = H//2
        # W = W//2
        # focal = focal/2.
        hwfs[:, 0] = hwfs[:, 0] // 2  # height
        hwfs[:, 1] = hwfs[:, 1] // 2  # width
        hwfs[:, 2] = hwfs[:, 2] / 2  # focal
        imgs = tf.image.resize_area(imgs, [int(hwfs[0,0]), int(hwfs[0,1])]).numpy()
        
    return imgs, names, poses, hwfs, shs, render_poses, i_split
