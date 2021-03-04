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
Results: The results has scan line noises, which is created by the predicted diffuse power and spherical harmonics power. Since the rendering is completed by batch, whcih is 2048 in the current setting, leading to scan line noises. 

----------------------------------------------------------------------------
Date: 2021/02/18
Experiments Index: 2
Detail: To solve this problem in 2021/02/18-1, we need to remove the predicted diffuse power and predicted spherical harmonics power. In stead we fine tune diffuse visibility map and local spherical harmonics with a static diffuse power say 5.0. So the current output should be albedo(3) + density(1) + normal(3) + specular(1) + diffuse_visibility(1) + sh(4), where albedo and normal can be ignored since we're using normal ground truth.
Logs: vgl-gpu04:./logs/*_local_sh
Commit: 0cf3deef083abe8133ec20675169a764190566c0
Results: Local spherical harmonics with gt albedo and gt normal generates the best quality ever. It seems the local spherical harmonics is able to generate scene dependent local lighting model. It's just the local spherical harmonics looks too strong, which can be decomposed into global spherical harmonics and local spherical harmonics

----------------------------------------------------------------------------
Date: 2021/02/19
Experiments Index: 1
Detail: Use gloabl spherical harmonics and local spherical harmonics to model the indirect lighting.
Logs: vgl-gpu04:./logs/*_sh_local_global
Commit: 6b79330e6ae85c503830bce96ca8776fe84ce4ec
Results: it seems global spherical harmonics and local spherical harmonics wont contribute twice. Use them once.

----------------------------------------------------------------------------
Date: 2021/02/23
Experiments Index: 1
Detail: Run experiments on real light stage data (Jing) and remove specular from rendering equation and also remove global spherical harmonics. This experiments is used to test if the provided normal map and albedo lacking eyeballs and teeth are able to generate reasonable results.
Logs: vgl-gpu04:./logs/*_local_sh_no_spec
Commit: d36a629cfd7d2689845dab9ab9901ea70ef37ac5
Results: looks good, but spherical harmonics lighting map looks a little strong

----------------------------------------------------------------------------
Date: 2021/02/23
Experiments Index: 2
Detail: The current diffuse light power is 4.0 and local shperical harmonics is 0.5. Let's try to adjust these two parameter and see the influence on the final results. Let's try 6.0 diffuse and 0.5 spherical harmonics
Logs: vgl-gpu04: ./logs/*_diffuse_6_0_sh_0_5
Commit: e08ba9f4d026586cc5c76ddd215c1107ea81d142
Results: It's clear that there's difference on the final rendering results, let's try if we disable the diffuse light, this will show all the shading components other than albedo

----------------------------------------------------------------------------
Date: 2021/02/24
Experiments Index: 1
Detail: Disable diffuse lighting and left with only local spherial harmonics lighting
Logs: vgl-gpu04:./logs/*_sh_only_0.5
Commit: b60160ffbce7d06b2f79c5252611a8eda7a3adf9
Results: Using local spherical harmonics only with 0.5 factor cannot lightup the scene properly. Let's see if the local spherical harmonics can lightup the entire scene.

----------------------------------------------------------------------------
Date: 2021/02/24
Experiments Index: 2
Detail: Remove 0.5 factor from the rendering equation
Commit: b6b01a281a0622017adb8ada1edeac74e6d7f647
Logs: vgl-gpu04:./logs/*_sh_only
Results: For a single light, using spherical harmonics only can model all lights and generate a realistic shading results, with a pretty good quality. The next thing to do is to run the lighting model on multiple lightings and test on unknown lighting.

----------------------------------------------------------------------------
Date: 2021/02/25
Experiments Index: 1
Detail: change the lit diffuse output to the multiplication of lit diffuse and diffuse visibility
Commit: 1a289204073453e5f2a745ad3359ef79a4adf7c2
Logs: vgl-gpu04:./logs/*_diffuse_times_vis
Results: the multiplication of lit diffuse (10 times diffuse lighting) and diffuse visibility generates a correct results compared with the past global spherical harmonics results. It's just the quality of diffuse visibility is not high resolution enough to be compatible with normal map and albedo map, which needs to be solved somwhow using normal map and albedo since the current diffuse visibility is a predicted visibility map generated from occupancy, which has been proved that nerf cannot generate a high accuracy occupancy field.

----------------------------------------------------------------------------
Date: 2021/02/26 -> 2021/03/01
Experiments Index: 1
Detail: make diffuse visibility high resolution. One possible way is to generate visibility map from gray scale image of albedo map, since the visibility of the current face should be related to it's albedo or normal. Let's try albedo first and normal then. Compare the results, see if we can improve sharpness. Using Matlab, it's clear that using z channel of normal could be the best, since z channel of the normal map can be viewed as normal light up from the front view.
Commit: e6d6299c7674408d7ff8f23b74557b98301c4824
Logs: vgl-gpu04:./logs/*_vis_normal_z
Results: The results looks good, the visiblity map looks good. Let's use z channel of the normal map to generate visibility map. Compared with the final results, the spherical harmonics looks a little strong than diffuse and also the rendered results lacking of specular map. Let's add specular also from normal z with specular term.

----------------------------------------------------------------------------
Date: 2021/03/02
Experiments Index: 1
Detail: Try lighter spherical harmonics lighting with maybe 0.5 as a factor and also generate a specular from normal z
Commit: dc78716336d1f361005f48a4b7036a2b555d2bb3
Logs: vgl-gpu04:./logs/*_spec_normal_z
Results: using 0.5 times spherical harmonics lighting and specular generates a results with obvious noise on the side face, which need to be figured out. Also, the results is not as good as using only previous experiments.

----------------------------------------------------------------------------
Date: 2021/03/03->2021/03/04
Experiments Index: 1
Detail: let's try to use litted diffuse light as specular.
Commit: ffe48a7fb6b9fd47a38a52bdf8f2fb964cc48ad0
Logs: vgl-gpu04:./logs/*_spec_lit
Results: 

----------------------------------------------------------------------------
Date: 2021/03/04
Experiments Index: 2
Detail: using jupyter notebook and check out if the local spherical harmonics has same values everywhere in the field.
Commit: 
Logs: Null
Results: 

----------------------------------------------------------------------------
Date: 2021/02/26
Experiments Index: 2
Detail: input global spherical harmonic
Commit: 
Logs: 
Results: 