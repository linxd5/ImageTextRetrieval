from model import ImgSenRanking
import cPickle as pkl
import torch
from datasets import load_dataset
import numpy as np
from tools import encode_sentences, encode_images
import json

data = 'arch'
loadfrom = 'vse/' + data
saveto = 'vse/%s_server/%s' %(data, data)
hyper_params = '%s_params.pkl'  % loadfrom
model_params = '%s_model.pkl' % loadfrom

print 'Building model ...   ',
model_options = pkl.load(open(hyper_params, 'r'))
model = ImgSenRanking(model_options).cuda()
model.load_state_dict(torch.load(model_params))
print 'Done'

test = load_dataset(data, load_test=True)

print 'Dumping data ...   '

curr_model = {}
curr_model['options'] = model_options
curr_model['worddict'] = model_options['worddict']
curr_model['word_idict'] = model_options['word_idict']
curr_model['img_sen_model'] = model

ls, lim = encode_sentences(curr_model, test[0]), encode_images(curr_model, test[1])

# save the using params and model when dumping data
torch.save(ls, '%s_ls.pkl'%saveto)
torch.save(lim, '%s_lim.pkl'%saveto)
pkl.dump(model_options, open('%s_params_dump.pkl'%saveto, 'wb'))
torch.save(model.state_dict(), '%s_model_dump.pkl'%saveto)
json.dump(test[0], open('%s_caps.json'%saveto, 'w'))

print 'ls: ', ls.data.size()
print 'lim: ', lim.data.size()

