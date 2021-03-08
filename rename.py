import os
import sys
import shutil

basedir = './logs'
dir_list = [
    'model_1_sh_21_test',
    'model_1_sh_24_test',
    'model_1_sh_32_test',
    'model_1_sh_37_test',
    'model_1_sh_38_test',
    'model_1_sh_45_test',
    'model_2_sh_21_test',
    'model_2_sh_24_test',
    'model_2_sh_32_test',
    'model_2_sh_37_test',
    'model_2_sh_38_test',
    'model_2_sh_45_test',
    'model_jing_sh_21_test', 
    'model_jing_sh_24_test', 
    'model_jing_sh_32_test', 
    'model_jing_sh_37_test', 
    'model_jing_sh_38_test', 
    'model_jing_sh_45_test', 
]
name = 'spec_lit'

for d in dir_list:
    # get path
    path_log = os.path.join(basedir, d)
    path_summary = os.path.join(basedir, 'summaries', d)

    path_log_new = os.path.join(basedir, '{}_{}'.format(d, name))
    path_summary_new = os.path.join(basedir, 'summaries', '{}_{}'.format(d, name))

    # rename path_log
    if os.path.exists(path_log):
        shutil.move(path_log, path_log_new)
        print(path_log, '->', path_log_new)
    else:
        print(path_log, 'is not existed')

    # rename path_summary
    if os.path.exists(path_summary):
        shutil.move(path_summary, path_summary_new)
        print(path_summary, '->', path_summary_new)
    else:
        print(path_summary, 'is not existed')