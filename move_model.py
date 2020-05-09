import os
import sys
model_type = 'output_data_join_utf8'
versions = [sys.argv[1]]

source_data = '/home_export/lat/%s/' % model_type
to_data = '/home_export/bzw/MRC/code/lic2020/results/%s/' % model_type

cmd_list = []
files = sorted((os.listdir(source_data)))
for file in files:
    for version in versions:
        if version in file and 'nbest_predictions' in file:
            print(file)
            
            #if 'test1' in file:
            #    division = 'test1'
            if 'dev' in file:
                division = 'dev'
            #elif 'train' in file:
            #    division = 'train'
            else:
                continue
                input('error! ')
            new_file = '{}_{}_nbest_predictions_utf8.json'.format(version, division)
            cmd = 'cp {} {}'.format(source_data+file, to_data+new_file)
            #cmd = 'scp {} bzw@10.108.218.217:{}'.format(source_data+file, to_data+new_file)
            print(cmd)
            flag = input('move?(空字符表示确认):')
            if flag != '':
                print('don\'t move! ')
            else:
#                 cmd_list.append(cmd)
                pro = os.popen(cmd)
                text = pro.read()
                print(text.strip())
