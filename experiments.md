----------------------------------------------------------------------------
Date: 2021/02/05
Experiments Index: 1
Detail: Try to have a baseline on gt albedo and gt normal, also gt imgs. The current nerf should includes model, model_fine, and model_material, where model_fine is used to generate rgb images only without using albedo and normal from model_material. This will generate the best quality using nerf for albedo, normal, as well as images estimation. If this experiment works, we should move on to connect albedo and normal from model_material to the model_fine and cancel the gradient back to model_fine.
Logs: logs_2/*_baseline_combine_density
Commit: bab2436cede175d63543b0277add93b8c27eda0c
Results: albedo looks not clear, also normal map looks not clear as well. RGB mpa looks correct. The reason could be the two network are currently using the same density value, which should be separate, using density from model_material to predict albedo as well as normal map and using density from model_fine to predict rgb

----------------------------------------------------------------------------
Date: 2021/02/06 -> 2021/02/07 (due to server reboot)
Experiments Index: 1
Detail: Use density from model_material to predict albedo as well as normal map and using density from model_fine to predict rgb and see if the predicted albedo and normal looks better.
Logs: ./logs/*_baseline_separate_density -> logs_2/*_baseline_separate_density
Commit: 083c2d089d0c39040ccbf4fe0ea540cdb3feee09
Results: albedo and normal map looks clear, which means they cannot share the same set of densities on the ray. It's recommended to apply a voxel based ground truth from 3D geometry, which is based on the inference of Nerf rendering equation, the network is actually learning the color of first intersection of the voxel grid.

----------------------------------------------------------------------------
Date: 2021/02/09
Experiments Index: 1
Detail: Hanyuan generated results with really high quality, which is applied with a mask on transparency channel. The implementation was on Pytorch here (https://github.com/CorneliusHsiao/nerf-relightable-pytorch). Check out commit 05849881b40a5828753501b3c964c87dcbccf332 in utils.py at Line 560.
Logs: /home/ICT2000/hxiao/now/logs/model_2_sh_21_test/version_57/tf
Commit: 05849881b40a5828753501b3c964c87dcbccf332 @ nerf-relightable-pytorch

----------------------------------------------------------------------------
Date: 2021/02/10 -> 2021/02/15 (due to new year)
Experiments Index: 1
Detail: The results look quite sharp and this experiment is used to implement normal mask on Tensorflow.
Logs: ./logs/*_alpha_mask
Commit: 78a116bba8704945166e43dd46105a220e23d9d7
Results: The results is indeed improved a lot on resolution and sharpness, however comparied with pytorch version from Hanyuan, the speed, memory, quality (displacement) seems not as good as the one from pytorch. We need to focus on the quality first, a 800*800 experiments need to be conducted. The implementation is confirmed, Pytorch version (with spherical harmonics input) has not only the alpha mask but also two more layer on the output.

----------------------------------------------------------------------------
Date: 2021/02/16
Experiments Index: 1
Detail: need a higher resolution results for details.
Logs: vgl-gpu03:./logs/*_alpha_mask_800
Commit: 
Results: It's obviously that there's no such big difference on training time for 400*400 and 800*800 cost on time complexity and space complexity. The current results look sharper than previous tf experiments, but still not as good as Hanyuan's.

----------------------------------------------------------------------------
Date: 2021/02/16
Experiments Index: 2
Detail: Since using alpha mask will help the network focus on face RGB prediction, this experiment is used to add normal loss as well as albedo loss.
Logs: vgl-gpu04:./logs/*_albedo_normal_loss
Commit: 1b96aada8fd0bbe095000dee404690ca1ec6c213
Results: Also, using mask improves the final results including albedo, normal, as well as rgb way better then experiments without mask, but still not sharp. Need to use ground truth directly.

----------------------------------------------------------------------------
Date: 2021/02/17
Experiments Index: 1
Detail: Use albedo and normal ground truth like alpha mask directly to the network output. For each ray, duplicate value from 2D to 3D.
Logs: vgl-gpu03:./logs/*_albedo_normal_gt
Commit: 1d1cb6481d30e282696efbcd377a29659c34e24d
Results: It's clear that the results is way sharper, but since all samples share the same albedo and normal along the same ray, the density is not working, leading to the misprediction of the weights along the ray and midprediction of the final color. Also, normal prediction is recommended, so that the network keeps the weights and memorize normal from multiview, which is vital for rendering.

----------------------------------------------------------------------------
Date: 2021/02/17
Experiments Index: 2
Detail: Use albedo ground truth directly, where details can be preserved, and let metwork predict normal with surpervision, so that the density can be preserved on each ray, which will not cause the midprediction on weights in experiment 2021/02/17-1.
Logs: vgl-gpu04:./logs/*_albedo_gt_normal_loss
Commit: ccb08170132a102dcdb9948f24e2a203482616db
Results: Solved the problem in 2021/02/17-1 of weights midprediction. and also the generated final rgb results can preserve details from albedo. Moreover, even though details on normal map is smoothed, it's still the best results ever. It's just the outputed diffuse light map is not enlarged by the predicted diffuse power, which should be corrected in the next experiment. If we take a closer look at supervised normal map, the loss is continuing degrade and resuting to a more detailed normal map then ever.

----------------------------------------------------------------------------
Date: 2021/02/18
Experiments Index: 1
Detail: Output diffuse light map, diffuse light map litted, spherical harmonics map, spherical harmoncis map litted. This is used to check if the diffuse light and spherical harmonics lighting looks correct, since previous outputed diffuse light map times 5.0, which is an arbitrary value.
Logs: vgl-gpu03:./logs/*_lit_diffuse_sh
Commit: 7cd3ac1ca3392caff23cb6ff0b344079fccb68c8
Results: 