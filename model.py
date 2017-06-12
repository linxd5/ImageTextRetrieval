# coding: utf-8
import torch
from utils import l2norm, xavier_weight
from torch.autograd import Variable
import torch.nn.init as init
from gensim.models.word2vec import Word2Vec
import numpy

wvModel = Word2Vec.load('static/word2vec-chi/word2vec_news.model')

class ImgSenRanking(torch.nn.Module):
    def __init__(self, model_options):
        super(ImgSenRanking, self).__init__()
        self.linear = torch.nn.Linear(model_options['dim_image'], model_options['dim'])
        self.lstm = torch.nn.LSTM(model_options['dim_word'], model_options['dim'], 1)
        self.embedding = torch.nn.Embedding(model_options['n_words'], model_options['dim_word'])
        self.model_options = model_options
        self.init_weights()

    def init_weights(self):
        xavier_weight(self.linear.weight)
        # init.xavier_normal(self.linear.weight)
        self.linear.bias.data.fill_(0)

    def forward(self, x_id, im, x):
        x_id_emb = self.embedding(x_id)
        im = self.linear(im)

        x_w2v = torch.zeros(*x_id_emb.size())
        x_cat = None
        if self.model_options['concat']:
            for i, text in enumerate(x):
                for j, word in enumerate(text.split()):
                    try:
                        x_w2v[j, i] = torch.from_numpy(wvModel[word.decode('utf8')])
                    except KeyError:
                        pass
            x_w2v = Variable(x_w2v.cuda())
            x_cat = torch.cat([x_id_emb, x_w2v])
        else:
            x_cat = x_id_emb


        if self.model_options['encoder'] == 'bow':
            x_cat = x_cat.sum(0).squeeze(0)
        else:
            _, (x_cat, _) = self.lstm(x_cat)
            x_cat = x_cat.squeeze(0)

        return l2norm(x_cat), l2norm(im)

    def forward_sens(self, x_id, x):
        x_id_emb = self.embedding(x_id)

        x_w2v = torch.zeros(*x_id_emb.size())
        x_cat = None
        if self.model_options['concat']:
            for i, text in enumerate(x):
                for j, word in enumerate(text):
                    try:
                        x_w2v[j, i] = torch.from_numpy(wvModel[word.decode('utf8')])
                    except KeyError:
                        pass

            x_w2v = Variable(x_w2v.cuda())
            x_cat = torch.cat([x_id_emb, x_w2v])
        else:
            x_cat = x_id_emb

        if self.model_options['encoder'] == 'bow':
            x_cat = x_cat.sum(0).squeeze(0)
        else:
            _, (x_cat, _) = self.lstm(x_cat)
            x_cat = x_cat.squeeze(0)
        return l2norm(x_cat)

    def forward_imgs(self, im):
        im = self.linear(im)
        return l2norm(im)

class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores)+scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()), (margin-diagonal).expand_as(scores).transpose(1, 0)+scores)

        for i in xrange(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return cost_s.sum() + cost_im.sum()
