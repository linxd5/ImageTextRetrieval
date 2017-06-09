# -*- coding: utf-8 -*-

import torch
import os
import cPickle as pkl
from datasets import load_dataset
from vocab import build_dictionary
import homogeneous_data
from torch.autograd import Variable
import time
from model import ImgSenRanking, PairwiseRankingLoss
import numpy
from tools import encode_sentences, encode_images
from evaluation import i2t, t2i, i2t_arch, t2i_arch

from hyperboard import Agent

cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

# @profile
def trainer(data='coco',
            margin=0.2,
            dim=1024,
            dim_image=4096,
            dim_word=300,
            encoder='gru',
            max_epochs=15,
            dispFreq=10,
            decay_c=0.0,
            grad_clip=2.0,
            maxlen_w=150,
            batch_size=128,
            saveto='vse/coco',
            validFreq=100,
            lrate=0.0002,
            concat=True,
            reload_=False):


    hyper_params = {
        'data': data,
        'encoder': encoder,
        'batch_size': batch_size,
        'time': cur_time,
        'lrate': lrate,
        'concat': concat,
    }

    i2t_r1 = dict([('i2t_recall', 'r1')]+hyper_params.items())
    i2t_r5 = dict([('i2t_recall', 'r5')]+hyper_params.items())
    i2t_r10 = dict([('i2t_recall', 'r10')]+hyper_params.items())
    t2i_r1 = dict([('t2i_recall', 'r1')]+hyper_params.items())
    t2i_r5 = dict([('t2i_recall', 'r5')]+hyper_params.items())
    t2i_r10 = dict([('t2i_recall', 'r10')]+hyper_params.items())

    i2t_med = dict([('i2t_med', 'i2t_med')]+hyper_params.items())
    t2i_med = dict([('t2i_med', 't2i_med')]+hyper_params.items())

    agent = Agent(port=5020)
    i2t_r1_agent = agent.register(i2t_r1, 'recall', overwrite=True)
    i2t_r5_agent = agent.register(i2t_r5, 'recall', overwrite=True)
    i2t_r10_agent = agent.register(i2t_r10, 'recall', overwrite=True)
    t2i_r1_agent = agent.register(t2i_r1, 'recall', overwrite=True)
    t2i_r5_agent = agent.register(t2i_r5, 'recall', overwrite=True)
    t2i_r10_agent = agent.register(t2i_r10, 'recall', overwrite=True)

    i2t_med_agent = agent.register(i2t_med, 'median', overwrite=True)
    t2i_med_agent = agent.register(t2i_med, 'median', overwrite=True)


    # Model options
    model_options = {}
    model_options['data'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_image'] = dim_image
    model_options['dim_word'] = dim_word
    model_options['encoder'] = encoder
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_
    model_options['concat'] = concat

    print model_options

    # reload options
    if reload_ and os.path.exists(saveto):
        print 'reloading...' + saveto
        with open('%s.pkl'%saveto, 'rb') as f:
            model_options = pkl.load(f)

    # Load training and development sets
    print 'loading dataset'
    train, dev = load_dataset(data)[:2]

    # Create and save dictionary
    print 'Create dictionary'
    worddict = build_dictionary(train[0]+dev[0])[0]
    n_words = len(worddict)
    model_options['n_words'] = n_words
    print 'Dictionary size: ' + str(n_words)
    with open('%s.dictionary.pkl'%saveto, 'wb') as f:
        pkl.dump(worddict, f)


    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    model_options['worddict'] = worddict
    model_options['word_idict'] = word_idict

    # Each sentence in the minibatch have same length (for encoder)
    train_iter = homogeneous_data.HomogeneousData([train[0], train[1]], batch_size=batch_size, maxlen=maxlen_w)

    img_sen_model = ImgSenRanking(model_options)
    img_sen_model = img_sen_model.cuda()

    loss_fn = PairwiseRankingLoss(margin=margin)
    loss_fn = loss_fn.cuda()

    params = filter(lambda p: p.requires_grad, img_sen_model.parameters())
    optimizer = torch.optim.Adam(params, lrate)

    uidx = 0
    curr = 0.0
    n_samples = 0

    for eidx in xrange(max_epochs):

        print 'Epoch ', eidx

        for x, im in train_iter:
            n_samples += len(x)
            uidx += 1

            x_id, im = homogeneous_data.prepare_data(x, im, worddict, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            x_id = Variable(torch.from_numpy(x_id).cuda())
            im = Variable(torch.from_numpy(im).cuda())
            # Update
            ud_start = time.time()
            x, im = img_sen_model(x_id, im, x)
            cost = loss_fn(im, x)
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm(params, grad_clip)
            optimizer.step()
            ud = time.time() - ud_start

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost.data.cpu().numpy()[0], 'UD ', ud

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                curr_model['worddict'] = worddict
                curr_model['word_idict'] = word_idict
                curr_model['img_sen_model'] = img_sen_model

                ls, lim = encode_sentences(curr_model, dev[0]), encode_images(curr_model, dev[1])

                r1, r5, r10, medr = 0.0, 0.0, 0.0, 0
                r1i, r5i, r10i, medri = 0.0, 0.0, 0.0, 0
                r_time = time.time()
                if data == 'arch' or data == 'arch_small':
                    (r1, r5, r10, medr) = i2t_arch(lim, ls)
                    print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
                    (r1i, r5i, r10i, medri) = t2i_arch(lim, ls)
                    print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)
                else:
                    (r1, r5, r10, medr) = i2t(lim, ls)
                    print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
                    (r1i, r5i, r10i, medri) = t2i(lim, ls)
                    print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)

                print "Cal Recall@K using %ss" %(time.time()-r_time)

                record_num = uidx / validFreq
                agent.append(i2t_r1_agent, record_num, r1)
                agent.append(i2t_r5_agent, record_num, r5)
                agent.append(i2t_r10_agent, record_num, r10)
                agent.append(t2i_r1_agent, record_num, r1i)
                agent.append(t2i_r5_agent, record_num, r5i)
                agent.append(t2i_r10_agent, record_num, r10i)

                agent.append(i2t_med_agent, record_num, medr)
                agent.append(t2i_med_agent, record_num, medri)

                currscore = r1 + r5 + r10 + r1i + r5i + r10i
                if currscore > curr:
                    curr = currscore

                    # Save model
                    print 'Saving model...',
                    pkl.dump(model_options, open('%s_params.pkl'%saveto, 'wb'))
                    torch.save(img_sen_model.state_dict(), '%s_model.pkl'%saveto)
                    print 'Done'

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass
