import numpy
import torch
from datasets import load_dataset
from tools import encode_sentences, encode_images
import json

def evalrank(model, data, split='dev'):
    """
    Evaluate a trained model on either dev ortest
    """

    print 'Loading dataset'
    if split == 'dev':
        X = load_dataset(data)[1]
    else:
        X = load_dataset(data, load_test=True)


    print 'Computing results...'
    ls = encode_sentences(model, X[0])
    lim = encode_images(model, X[1])

    if data == 'arch':
        # Find the good case in test dataset
        (r1, r5, r10, medr) = i2t_arch_case(lim, ls, X[0])
        print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
        (r1i, r5i, r10i, medri) = t2i_arch_case(lim, ls, X[0])
        print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)
    else:
        (r1, r5, r10, medr) = i2t(lim, ls)
        print "Image to text: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr)
        (r1i, r5i, r10i, medri) = t2i(lim, ls)
        print "Text to image: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri)


def i2t(images, captions, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] / 5

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5*index, 5*index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i(images, captions, npts=None, data='f8k'):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] / 5

    ims = torch.cat([images[i].unsqueeze(0) for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5*index : 5*index + 5]

        # Compute scores
        d = torch.mm(queries, ims.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[5 * index + i] = numpy.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def i2t_arch(images, captions):
    npts = images.size()[0]
    ranks = numpy.zeros(npts)
    caps_obj_id = numpy.load(open('data/arch/arch_dev_caps_id.npy'))
    imgs_obj_id = numpy.load(open('data/arch/arch_dev_imgs_id.npy'))
    for index in range(npts):
        # Get query image
        im = images[index:index+1]
        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()
        ranks[index] = numpy.where(caps_obj_id[inds] == imgs_obj_id[index])[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i_arch(images, captions):
    npts = captions.size()[0]
    ranks = numpy.zeros(npts)
    caps_obj_id = numpy.load(open('data/arch/arch_dev_caps_id.npy'))
    imgs_obj_id = numpy.load(open('data/arch/arch_dev_imgs_id.npy'))
    for index in range(npts):
        # Get query caption
        cap = captions[index:index+1]
        # Compute scores
        d = torch.mm(cap, images.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()
        ranks[index] = numpy.where(imgs_obj_id[inds] == caps_obj_id[index])[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)

def i2t_arch_case(images, captions, caps_orig):
    npts = images.size()[0]
    ranks = numpy.zeros(npts)
    caps_obj_id = numpy.load(open('data/arch/arch_test_caps_id.npy'))
    imgs_obj_id = numpy.load(open('data/arch/arch_test_imgs_id.npy'))
    imgs_url = json.load(open('data/arch/arch_test_imgs_url.json'))

    print_num = 10
    for index in range(npts):
        # Get query image
        im = images[index:index+1]
        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()
        ranks[index] = numpy.where(caps_obj_id[inds] == imgs_obj_id[index])[0][0]
        temp_rank = int(ranks[index])
        if temp_rank == 0 and print_num > 0:
            print 'i2t:  %d' %(10-print_num)
            print 'image_url: ', imgs_url[index]
            print 'captions ',  caps_orig[inds[0]]
            print '\n\n'
            print_num -= 1

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i_arch_case(images, captions, caps_orig):
    npts = captions.size()[0]
    ranks = numpy.zeros(npts)
    caps_obj_id = numpy.load(open('data/arch/arch_test_caps_id.npy'))
    imgs_obj_id = numpy.load(open('data/arch/arch_test_imgs_id.npy'))
    imgs_url = json.load(open('data/arch/arch_test_imgs_url.json'))
    print_num = 10
    for index in range(npts):
        # Get query caption
        cap = captions[index:index+1]
        # Compute scores
        d = torch.mm(cap, images.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()
        ranks[index] = numpy.where(imgs_obj_id[inds] == caps_obj_id[index])[0][0]
        temp_rank = int(ranks[index])
        if temp_rank == 0 and print_num > 0:
            print 't2i:  %d' %(10-print_num)
            print 'caption: ', caps_orig[index]
            print 'img_url: ', imgs_url[inds[0]]
            print '\n\n'
            print_num -= 1

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)
