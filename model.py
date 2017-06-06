import torch
from utils import l2norm
from torch.autograd import Variable

class jianzhuNet(torch.nn.Module):
    def __init__(self, model_options):
        super(jianzhuNet, self).__init__()
        self.image_fc = torch.nn.Linear(model_options['dim_image'], model_options['dim'])
        self.rnn = torch.nn.LSTM(model_options['dim_word'], model_options['dim'], 1)
        self.embedding = torch.nn.Embedding(model_options['n_words'], model_options['dim_word'])
        self.model_options = model_options

    def forward(self, x, im):
        x = self.embedding(x)
        im = self.image_fc(im)

        if self.model_options['encoder'] == 'bow':
            x = x.sum(0).squeeze(0)
        else:
            _, (x, _) = self.rnn(x)
            x = x.squeeze(0)

        return l2norm(x), l2norm(im)

    def forward_sens(self, x):
        x = self.embedding(x)
        if self.model_options['encoder'] == 'bow':
            x = x.sum(0).squeeze(0)
        else:
            _, (x, _) = self.rnn(x)
            x = x.squeeze(0)
        return l2norm(x)

    def forward_imgs(self, im):
        im = self.image_fc(im)
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
