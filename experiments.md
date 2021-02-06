----------------------------------------------------------------------------
Date: 2021/02/05
Experiments Index: 1
Detail: Try to have a baseline on gt albedo and gt normal, also gt imgs. The current nerf should includes model, model_fine, and model_material, where model_fine is used to generate rgb images only without using albedo and normal from model_material. This will generate the best quality using nerf for albedo, normal, as well as images estimation. If this experiment works, we should move on to connect albedo and normal from model_material to the model_fine and cancel the gradient back to model_fine.
Logs: logs_2/*_baseline
Commit: 
Results: 