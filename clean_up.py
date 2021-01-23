import os
import sys
import shutil

dir_sh_list = [
    # './logs/model_1_sh_21_test',
    # './logs/model_1_sh_24_test',
    # './logs/model_1_sh_32_test',
    # './logs/model_1_sh_37_test',
    './logs/model_1_sh_38_test',
    # './logs/model_1_sh_45_test',
]

summary_sh_list = [
    # './logs/summaries/model_1_sh_21_test',
    # './logs/summaries/model_1_sh_24_test',
    # './logs/summaries/model_1_sh_32_test',
    # './logs/summaries/model_1_sh_37_test',
    './logs/summaries/model_1_sh_38_test',
    # './logs/summaries/model_1_sh_45_test',
]

# dir_sh_list = [
#     './logs/model_1_partial_test',
# ]

# summary_sh_list = [
#     './logs/summaries/model_1_partial_test',
# ]

# dir_sh_list = [
#     './logs/nerf_lego_all_test',
#     './logs/nerf_lego_partial_test',
#     './logs/nerf_hotdog_test',
#     './logs/nerf_chair_test',
#     './logs/nerf_materials_test',
#     './logs/nerf_ficus_test',
# ]

# summary_sh_list = [
#     './logs/summaries/nerf_lego_all_test',
#     './logs/summaries/nerf_lego_partial_test',
#     './logs/summaries/nerf_hotdog_test',
#     './logs/summaries/nerf_chair_test',
#     './logs/summaries/nerf_materials_test',
#     './logs/summaries/nerf_ficus_test',
# ]

for d in dir_sh_list:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(d, 'is removed')
    else:
        print(d, 'is not existed')

for d in summary_sh_list:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(d, 'is removed')
    else:
        print(d, 'is not existed')