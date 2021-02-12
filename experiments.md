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
Logs: logs_2/*_baseline_separate_density
Commit: 083c2d089d0c39040ccbf4fe0ea540cdb3feee09
Results: albedo and normal map looks clear, which means they cannot share the same set of densities on the ray. It's recommended to apply a voxel based ground truth from 3D geometry, which is based on the inference of Nerf rendering equation, the network is actually learning the color of first intersection of the voxel grid.

----------------------------------------------------------------------------
Date: 2021/02/09 (due to server reboot)
Experiments Index: 1
Detail: Hanyuan generated results with really high quality, which is applied with a mask on transparency channel. The implementation was on Pytorch here (https://github.com/CorneliusHsiao/nerf-relightable-pytorch). Check out commit 05849881b40a5828753501b3c964c87dcbccf332 in utils.py at Line 560. The results look quite sharp and this experiment is used to implement normal mask on Tensorflow.
Logs: logs_2/*_alpha_mask
Commit: 
Results: 