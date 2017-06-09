# -*- coding: utf-8 -*-
import os
import json
import random

dev_num = 10000

data_pair = []

# Keep pace with coco dataset in
# - json field : caption, img_path, obj_id
# - dataset split : train/val=2/1

# Feel free to choose title or detail as caption

with open('jianzhu_tag.json') as f_read:
    source_data = map(json.loads, f_read.readlines())

    for k, line in enumerate(source_data):
        if k % 1000 == 0:
            print 'Processing %d / %d' %(k, len(source_data))
        detail = line['detail'].strip()
        title, tag = line['title'].strip(), line['tag'].strip()
        title = title if title != '' else tag
        url = line['other']['url'] if line['other'].has_key('url') else ''
        poster = line['poster'].replace('/data/crawler/', '')
        if poster == '':
            continue
        poster = os.path.join('../../', poster)

        if os.path.exists(poster):
            for img_name in os.listdir(poster):
                poster = poster.replace('../../', 'static/dataset/')
                img_path = os.path.join(poster, img_name)
                data_pair.append({'caption': detail, 'img_path': img_path,
                                  'obj_id': k, 'url': url})

random.shuffle(data_pair)
print 'Remaining data_pair: ', len(data_pair)

json.dump(data_pair[:dev_num], open('data_pair_val.json', 'w'))
json.dump(data_pair[dev_num:], open('data_pair_train.json', 'w'))
