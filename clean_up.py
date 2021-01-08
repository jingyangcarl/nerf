import os
import sys
import shutil

dir_sh_list = [
    './logs/model_0_sh_21_test',
    './logs/model_0_sh_24_test',
    './logs/model_0_sh_32_test',
    './logs/model_0_sh_37_test',
    './logs/model_0_sh_38_test',
    './logs/model_0_sh_45_test',
]

summary_sh_list = [
    './logs/summaries/model_0_sh_21_test',
    './logs/summaries/model_0_sh_24_test',
    './logs/summaries/model_0_sh_32_test',
    './logs/summaries/model_0_sh_37_test',
    './logs/summaries/model_0_sh_38_test',
    './logs/summaries/model_0_sh_45_test',
]

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