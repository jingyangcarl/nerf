import os
import tensorflow as tf
import numpy as np
import imageio
import json


def trans_t(t): return tf.convert_to_tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1],
], dtype=tf.float32)


def rot_phi(phi): return tf.convert_to_tensor([
    [1, 0, 0, 0],
    [0, tf.cos(phi), -tf.sin(phi), 0],
    [0, tf.sin(phi), tf.cos(phi), 0],
    [0, 0, 0, 1],
], dtype=tf.float32)


def rot_theta(th): return tf.convert_to_tensor([
    [tf.cos(th), 0, -tf.sin(th), 0],
    [0, 1, 0, 0],
    [tf.sin(th), 0, tf.cos(th), 0],
    [0, 0, 0, 1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0],
                    [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_lightstage_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    # list initialization for all
    all_imgs = []
    all_lightProbes = []
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
        lightProbes = []
        names = []
        poses = []
        hwfs = []
        shs = []

        # set skip parameter
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        # read each frame
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            lightProbes.append(imageio.imread(os.path.join(basedir, frame['lightProbe_path'] + '.exr')))
            names.append(os.path.splitext(os.path.basename(fname))[0])
            poses.append(np.array(frame['transform_matrix']))
            hwfs.append(np.array(frame['hwf']))
            shs.append(np.array(frame['sh']))

        # type process
        # keep all 4 channels (RGBA)
        imgs = (np.array(imgs) / 255.).astype(np.float32)
        lightProbes = np.array(lightProbes).astype(np.float32)
        names = np.array(names).astype(np.str)
        poses = np.array(poses).astype(np.float32)
        hwfs = np.array(hwfs).astype(np.float32)
        shs = np.array(shs).astype(np.float32)

        # prepare for serialization
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_lightProbes.append(lightProbes)
        all_names.append(names)
        all_poses.append(poses)
        all_hwfs.append(hwfs)
        all_shs.append(shs)

    # split index
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    # serialization
    imgs = np.concatenate(all_imgs, 0)
    lightProbes = np.concatenate(all_lightProbes, 0)
    names = np.concatenate(all_names, 0)
    poses = np.concatenate(all_poses, 0)
    hwfs = np.concatenate(all_hwfs, 0)
    shs = np.concatenate(all_shs, 0)

    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)

    # generate novel render poses
    render_poses = tf.stack([pose_spherical(angle, -10.0, 200.0)
                             for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        # H = H//2
        # W = W//2
        # focal = focal/2.
        hwfs[:, 0] = hwfs[:, 0] // 2  # height
        hwfs[:, 1] = hwfs[:, 1] // 2  # width
        hwfs[:, 2] = hwfs[:, 2] / 2  # focal
        imgs = tf.image.resize_area(imgs, [int(hwfs[0,0]), int(hwfs[0,1])]).numpy()

    return imgs, lightProbes, names, poses, hwfs, shs, render_poses, i_split
