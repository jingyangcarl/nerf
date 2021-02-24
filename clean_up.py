import os
import sys
import shutil

basedir = './logs'
dir_list = [
    # 'model_1_sh_21_test',
    # 'model_1_sh_24_test',
    # 'model_1_sh_32_test',
    # 'model_1_sh_37_test',
    # 'model_1_sh_38_test',
    # 'model_1_sh_45_test',
    # 'model_2_sh_21_test',
    # 'model_2_sh_24_test',
    # 'model_2_sh_32_test',
    # 'model_2_sh_37_test',
    # 'model_2_sh_38_test',
    # 'model_2_sh_45_test',
    'model_jing_sh_21_test', 
    'model_jing_sh_24_test', 
    'model_jing_sh_32_test', 
    'model_jing_sh_37_test', 
    'model_jing_sh_38_test', 
    'model_jing_sh_45_test', 
]

for d in dir_list:
    # get path
    path_log = os.path.join(basedir, d)
    path_summary = os.path.join(basedir, 'summaries', d)

    # remove path_log
    if os.path.exists(path_log):
        shutil.rmtree(path_log)
        print(path_log, 'is removed')
    else:
        print(path_log, 'is not existed')

    # remote path_summary
    if os.path.exists(path_summary):
        shutil.rmtree(path_summary)
        print(path_summary, 'is removed')
    else:
        print(path_summary, 'is not existed')