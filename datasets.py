"""
Dataset loading
"""
import numpy

path_to_data = 'data/'

def load_dataset(name='f8k', load_train=True):
    """
    Load captions and image features
    """
    loc = path_to_data + name + '/'

    # Captions
    train_caps, dev_caps, test_caps = [], [], []
    test_ims = []
    if load_train:
        with open(loc+name+'_train_caps.txt', 'rb') as f:
            for line in f:
                train_caps.append(line.strip())
    else:
        train_caps = None

    with open(loc+name+'_dev_caps.txt', 'rb') as f:
        for line in f:
            dev_caps.append(line.strip())

    # TODO: Add arch test data
    if name != 'arch':
        with open(loc+name+'_test_caps.txt', 'rb') as f:
            for line in f:
                test_caps.append(line.strip())
        test_ims = numpy.load(loc+name+'_test_ims.npy')

    # Image features
    if load_train:
        train_ims = numpy.load(loc+name+'_train_ims.npy')
    else:
        train_ims = None
    dev_ims = numpy.load(loc+name+'_dev_ims.npy')

    if name != 'arch':
        return (train_caps, train_ims), (dev_caps, dev_ims), (test_caps, test_ims)
    else:
        return (train_caps, train_ims), (dev_caps, dev_ims)