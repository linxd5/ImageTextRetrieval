# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import torch.nn
from torch.autograd import Variable
from model import ImgSenRanking
from PIL import Image, ImageFile
from flask import Flask, request, render_template, jsonify
from tools import encode_sentences, encode_images
from pre_transforms import image_transform, resnet
import cPickle as pkl
import torch
# TODO: Defind text_transforms in pre_transforms.py
import jieba.analyse
jieba.analyse.set_stop_words('static/dataset/stopwords.txt')

app = Flask(__name__)


ImageFile.LOAD_TRUNCATED_IMAGES = True

UPLOAD_FOLDER = 'static/upload/'
dump_path = 'vse/arch_server/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print 'loading image_dump.json'
images_dump = torch.load(os.path.join(dump_path, 'arch_lim.pkl'))
images_path = json.load(open(os.path.join(dump_path, 'arch_test_imgs_path.json')))
images_url = json.load(open(os.path.join(dump_path, 'arch_test_imgs_url.json')))

print 'loading text_dump.json'
texts_dump = torch.load(os.path.join(dump_path, 'arch_ls.pkl'))
texts_orig = json.load(open(os.path.join(dump_path, 'arch_caps.json')))
texts_url = json.load(open(os.path.join(dump_path, 'arch_test_caps_url.json')))

print 'loading jianzhu model'
model_options = pkl.load(open(os.path.join(dump_path, 'arch_params_dump.pkl')))
model = ImgSenRanking(model_options).cuda()
model.load_state_dict(torch.load(os.path.join(dump_path, 'arch_model_dump.pkl')))

curr_model = {}
curr_model['options'] = model_options
curr_model['worddict'] = model_options['worddict']
curr_model['word_idict'] = model_options['word_idict']
curr_model['img_sen_model'] = model


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_sen = request.form.get('query_sentence', '')
    k_input = int(request.form.get('k_input', ''))
    query_img = request.files['query_image']
    img_name = query_img.filename
    upload_img = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    sim_images, sim_images_url = [], []
    sim_texts, sim_texts_url = [], []
    if img_name:
        query_img.save(upload_img)
        img_vec = image_transform(Image.open(upload_img).convert('RGB')).unsqueeze(0)
        image_emb = encode_images(curr_model, resnet(Variable(img_vec.cuda())).data.cpu().numpy())
        d = torch.mm(image_emb, texts_dump.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()
        # sim_text_degree = 1-distance[0][:k_input]/distance[0][-1]
        sim_texts = np.array(texts_orig)[inds[:k_input]]
        sim_texts_url = np.array(texts_url)[inds[:k_input]]
        # sim_texts, sim_text_degree = sim_texts.tolist(), sim_text_degree.tolist()
        sim_texts, sim_texts_url = sim_texts.tolist(), sim_texts_url.tolist()
    if query_sen:
        query_sen = ' '.join(jieba.analyse.extract_tags(query_sen, topK=100, withWeight=False, allowPOS=()))
        query_sen = [query_sen.encode('utf8')]
        sentence = encode_sentences(curr_model, query_sen)
        d = torch.mm(sentence, images_dump.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()
        # sim_image_degree = 1-distance[0][:k_input]/distance[0][-1]
        sim_images = np.array(images_path)[inds[:k_input]]
        sim_images_url = np.array(images_url)[inds[:k_input]]
        # sim_images, sim_image_degree = sim_images.tolist(), sim_image_degree.tolist()
        sim_images, sim_images_url = sim_images.tolist(), sim_images_url.tolist()

    upload_img = upload_img if img_name else 'no_upload_img'
    return jsonify(sim_images=sim_images, sim_images_url=sim_images_url,
                   upload_img=upload_img, sim_texts=sim_texts, sim_texts_url=sim_texts_url)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2333)
