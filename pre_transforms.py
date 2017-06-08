
# coding: utf-8

# In[1]:

import jieba.analyse
jieba.analyse.set_stop_words('static/dataset/stopwords.txt')

from torchvision import transforms
import json, os
import numpy as np
import torch
from PIL import Image, ImageFile
import torchvision.models as models
from torch.autograd import Variable

dataset_dir = 'static/dataset/arch'
max_num = 1000000

# In[2]:

image_transform = transforms.Compose([
    transforms.Scale([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
])

resnet = models.resnet152(pretrained=True)
resnet.fc = torch.nn.Dropout(p=0)
resnet = resnet.eval()
resnet = resnet.cuda()


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm


def pre_transforms():

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print 'Pre-transforming train ...'

    json_file = 'annotations/data_pair_train.json'
    train_data_pair = json.load(open(os.path.join(dataset_dir, json_file), 'r'))

    captions_train, images_train = [], []
    for k, item in enumerate(train_data_pair):
        if k % 100 == 0:
            print 'Processing %d/%d' %(k, len(train_data_pair))
        caption = ' '.join(jieba.analyse.extract_tags(item['caption'], topK=100, withWeight=False, allowPOS=()))
        if len(caption) == 0:
            continue

        captions_train.append(caption.encode('utf8')+'\n')

        img_vec = image_transform(Image.open(item['img_path']).convert('RGB')).unsqueeze(0)
        img_vec = resnet(Variable(img_vec.cuda())).data.squeeze(0).cpu().numpy()
        images_train.append(img_vec)

        if k > max_num:
            break

    with open('arch_train_caps.txt', 'w') as f_write:
        f_write.writelines(captions_train)

    images_train = np.asarray(images_train, dtype=np.float32)
    images_train = normalize(images_train)
    np.save('arch_train_ims.npy', images_train)

    print 'Pre-transforming train Done'

    print 'Pre-transforming dev ...'

    json_file = 'annotations/data_pair_val.json'
    dev_data_pair = json.load(open(os.path.join(dataset_dir, json_file), 'r'))

    captions_dev, images_dev = [], []
    caps_obj_id, imgs_obj_id = [], []
    caps_url, imgs_url, imgs_path = [], [], []
    for k, item in enumerate(dev_data_pair):
        if k % 100 == 0:
            print 'Processing %d/%d' %(k, len(dev_data_pair))
        if item['obj_id'] not in caps_obj_id:
            caption = ' '.join(jieba.analyse.extract_tags(item['caption'], topK=100, withWeight=False, allowPOS=()))
            if len(caption) == 0:
                continue

            captions_dev.append(caption.encode('utf8')+'\n')
            caps_obj_id.append(item['obj_id'])
            caps_url.append(item['url'])

        img_vec = image_transform(Image.open(item['img_path']).convert('RGB')).unsqueeze(0)
        img_vec = resnet(Variable(img_vec.cuda())).data.squeeze(0).cpu().numpy()
        images_dev.append(img_vec)
        imgs_obj_id.append(item['obj_id'])
        imgs_url.append(item['url'])
        imgs_path.append(item['img_path'])

        if k > max_num:
            break

    with open('arch_dev_caps.txt', 'w') as f_write:
        f_write.writelines(captions_dev)

    json.dump(caps_url, open('arch_dev_caps_url.json', 'w'))
    json.dump(imgs_url, open('arch_dev_imgs_url.json', 'w'))
    json.dump(imgs_path, open('arch_dev_imgs_path.json', 'w'))

    images_dev = np.asarray(images_dev, dtype=np.float32)
    images_dev = normalize(images_dev)
    np.save('arch_dev_ims.npy', images_dev)

    caps_obj_id = np.asarray(caps_obj_id, dtype=np.float32)
    imgs_obj_id = np.asarray(imgs_obj_id, dtype=np.float32)
    np.save('arch_dev_caps_id.npy', caps_obj_id)
    np.save('arch_dev_imgs_id.npy', imgs_obj_id)

    print 'Pre-transforming dev Done'

    print 'Pre-transforming test ...'

    test_data_pair = train_data_pair + dev_data_pair

    captions_test, images_test = [], []
    caps_obj_id_test, imgs_obj_id_test = [], []
    caps_url_test, imgs_url_test, imgs_path_test = [], [], []
    for k, item in enumerate(test_data_pair):
        if k % 100 == 0:
            print 'Processing %d/%d' %(k, len(test_data_pair))
        if item['obj_id'] not in caps_obj_id_test:
            caption = ' '.join(jieba.analyse.extract_tags(item['caption'], topK=100, withWeight=False, allowPOS=()))
            if len(caption) == 0:
                continue

            captions_test.append(caption.encode('utf8')+'\n')
            caps_obj_id_test.append(item['obj_id'])
            caps_url_test.append(item['url'])

        img_vec = image_transform(Image.open(item['img_path']).convert('RGB')).unsqueeze(0)
        img_vec = resnet(Variable(img_vec.cuda())).data.squeeze(0).cpu().numpy()
        images_test.append(img_vec)
        imgs_obj_id_test.append(item['obj_id'])
        imgs_url_test.append(item['url'])
        imgs_path_test.append(item['img_path'])

        if k > max_num:
            break

    with open('arch_test_caps.txt', 'w') as f_write:
        f_write.writelines(captions_test)

    json.dump(caps_url_test, open('arch_test_caps_url.json', 'w'))
    json.dump(imgs_url_test, open('arch_test_imgs_url.json', 'w'))
    json.dump(imgs_path_test, open('arch_test_imgs_path.json', 'w'))

    images_test = np.asarray(images_test, dtype=np.float32)
    images_test = normalize(images_test)
    np.save('arch_test_ims.npy', images_test)

    caps_obj_id_test = np.asarray(caps_obj_id_test, dtype=np.float32)
    imgs_obj_id_test = np.asarray(imgs_obj_id_test, dtype=np.float32)
    np.save('arch_test_caps_id.npy', caps_obj_id_test)
    np.save('arch_test_imgs_id.npy', imgs_obj_id_test)

    print 'Pre-transforming test Done'

if __name__ == "__main__":
    pre_transforms()
