import os.path as osp
import glob
import random

data_dir = 'data/market1501/raw/Market-1501-v15.09.15/bounding_box_train/'
output_path = 'market_train_list.txt'
shuffle_times = 2000
random.seed(2018)

imglist = [osp.basename(f) for f in glob.glob(osp.join(data_dir, '*.jpg'))]
id_to_imgfile_dict = {}
for filename in imglist:
    pid = filename.split('_')[0]
    if pid not in id_to_imgfile_dict.keys():
        id_to_imgfile_dict[pid] = [filename]
    else:
        id_to_imgfile_dict[pid].append(filename)
all_ids = list(id_to_imgfile_dict.keys())
        
msg = ''
for _ in range(shuffle_times):
    random.shuffle(all_ids)
    for pid in all_ids:
        for filename in id_to_imgfile_dict[pid]:
            msg += '{}\n'.format(filename)

with open(output_path, 'w') as f:
    f.write(msg)
