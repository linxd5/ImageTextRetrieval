


## 关于项目的几个约定

1. 这篇文档用来详细地介绍**图文互搜项目**每天遇到的问题。这些信息经过提炼分类（例如 Python、PyTorch、Numpy等）后分别归档到相应的为知笔记中。

2. 对项目进行 git 版本管理是为了厘清每个阶段都做了什么事情，写了多少代码。最好每写完一个小模块就提交一次代码。同时，为了避免不必要的烦扰，只在本地电脑建立并维护项目 git，然后通过 PyCharm 将代码同步到服务器。

3. 临时的 ipython notebook 测试代码最好放在R项目外面。有些固定的 ipython notebook 代码也可以放在项目中，例如数据预处理代码，数据可视化代码。


---

## 项目代码详解

1. dump_data_pair.py：从原始数据 jianzhu_tag.json 中抽取特征 title, detail, image, (url)，同时对数据做 shuffle.

2. pre_transforms.py：为了加快模型训练的过程，对数据做预处理。包括图片部分的打开图片、图片预处理（剪切、归一化等）、使用预训练模型得到图片特征向量，文本部分的结巴抽取关键词、转换为词向量。预训练模型要使用 evaluation 模式，处理后的数据提供了 json 和 pymongo 两种存储方案。

3. datautil_pymongo.py：(1) 模型训练前的数据读取。包括划分训练集和验证集、构造负例、RNN模型输入的处理等。(2) 模型查询前的数据读取。包括图片数据读取和文本数据读取。

4. main.py：从 model.py 中读取模型，从 datautil_pymongo.py 中读取数据，然后做模型的训练、验证和效果评测(recall@k).

5. query_dump.py：项目查询前的数据准备。




## 进度报告

<font color='red'>argsort 要认真研究，谨慎使用！</font>


**图文互搜是不是一个伪命题？** 用图片搜索文本，我们可以先使用图片搜索图片，在找到图片对应的文本。 因为同一个 modal 里做搜索效果应该会比不同 modal 间做搜索好。既然我们不做图片描述的生成，而只是搜索图片描述，那我们就没有必要使用图文互搜。


Recall@K 的计算可否改成 GPU?  <font color='red'>很难！！</font> 可以考虑将 validFreq 调小。

ipython 中使用 `torch.cuda.set_device(1)` 可以设置程序在第一块 gpu 上运行。

pytorch 的 np.where 的接近实现 https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/3 。

找不到 good case，在计算 Recall@K 的时候考虑输出 good case.


#### Image Sentence Retrieval 相关论文和代码

- Order-embeddings of Images and Language.  Code (theano): https://github.com/ivendrov/order-embedding
- Multimodal Convolutional Neural Networks for Matching Image and Sentence. Code (Keras): https://github.com/jazzsaxmafia/m_CNN


#### Attention

**Code:**
- [yunjey / show-attend-and-tell](https://github.com/yunjey/show-attend-and-tell)
- [kelvinxu / arctic-captions](https://github.com/kelvinxu/arctic-captions)
- [szagoruyko / attention-transfer](https://github.com/szagoruyko/attention-transfer)

**Paper:**
- [Convolutional Sequence to Sequence Learning](https://zhuanlan.zhihu.com/p/26985192)
- [Paying more attention to attention](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1612.03928.pdf)



#### Python 3.6 项目迁移 (96 服务器 without sudo 权限)

1. Python 官网下载或者同事目录下拷贝 python3.6 并在本地安装
```Python
wget https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz
tar -zxvf Python-3.6.1.tgz
cd Python-3.6.1
./configure --enable-shared --with-ssl --prefix=/home/lindayong/bin/python36
make
make install
```
2. 设置 python3.6 的共享文件路径和 python3.6 的永久别名： 在.zshrc文件中加入
```Python
>> vim ~/.zshrc
alias python3.6='/home/lindayong/bin/python36/bin/python3.6'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/lindayong/bin/python36/lib
>> source ~/.zshrc
```

3. 创建虚拟环境并安装 pytorch 和 torchvision
```Python
mkdir env_py3.6
cd env_py3.6
python3.6 -m venv .
source bin/activate
pip install torch-0.1.11.post5-cp36-cp36m-linux_x86_64.whl
pip install torchvision
```

4. 卸载 ipython(2.7) 安装 ipython3.6. ipython notebook 需要重新安装




#### Python 字符编码

昨天两次遇到 Python 字符编码的问题。一次是在 pre_transforms 数据预处理的时候，得到的 caption 必须要 **encode('utf8')** 才能存储成文件。另一次是在 tools 查看某个词语是否在字典中，需要将该词语 w **decode('utf8')** 才能和字典中的 keys 做比较！

encode() 和 decode() 主要涉及 str 和 unicode 之间的变换。第一种问题遇到的话，会显式报错。但第二种问题遇到的时候，就很难被发现了，我也是dubug了一天才发现这个问题。所以教训告诉我们，**快使用 python3 吧！！**


Python 有两种字符串类型，str 和 unicode，我们可以通过 type(a) 来查看字符串 a 属于哪种类型。

所谓的编码方式，是对 word 编码的一种约定。比如某个 word 要用多少个字节编码，编码成什么字节。用了这种约定之后，我们才可以准确地保存和读取存储在文件中数据。不同的编码方式会有不同的约定，譬如 UTF8、UTF16、ASCII、GBK、GB2312 等等。对于不同编码方式的数据，我们是处理不了的，因为它们互相之间都不认识。这时候应该怎么办呢？ 幸运的是，我们可以**通过 decode() 告诉 Python 某个数据是使用什么方式编码的，然后 Python 才能准确地读取数据，并将其转换成 Python 统一的 Unicode 编码。**这个过程类似于我要把中文的"一"和英文的"two"相加的话，我要把他们转换成统一的阿拉伯数字，分别是 1 和 2，然后才能将二者相加。<font color='red'>打开文件时，如果没有告诉 Python 该文件的编码方式，Ubuntu Python2.7 默认该文件使用 UTF8 编码？</font>

下面来看几个 Python2.7 字符编码的错误样例。第一个样例出现在 pre_transforms.py 中。

```Python
# caption type: unicode
captions_train.append(caption+'\n')
with open('arch_train_caps.txt', 'w') as f_write:
    f_write.writelines(captions_train)
```
这里出现的错误是 `TypeError: writelines() argument must be a sequence of strings`。 原因是 caption 的类型是 unicode，加上 str 后类型仍然是 unicode。f_write.writelines 不支持 unicode 的直接写入，必须通过 encode() 将数据转换成某个编码格式的 str，才能写入到文件中，代码如下：
```Python
# caption type: unicode
captions_train.append(caption.encode('utf8')+'\n')
...
```
**json.dump** 支持 unicode 和 str 数据的直接写入。这里不适用 json.dump 是为了在 load_dataset 时和 VSE theano 数据保持一致。





#### pip 配置文件

在 ~/.pip/ 目录下创建 pip.conf 文件，并进行如下配置：
```python
[global]
time-out = 60
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```


#### 效果分析

验证集上 Recall@K 的计算结果如下：
- Image to text: 19.3 / 33.0 / 39.3 / 26.0
- Text to image: 29.6 / 46.5 / 52.2 / 8.0

但在查询的时候，图片搜索文本的效果可以到达 Image to text Recall@K 的效果，但是文本搜索图片的效果达不到 Text to image Recall@K 的效果。原因是<font color='red'>图片向量可以搜索相似图片向量，但文本向量没有办法搜索相似文本向量。</font> text_sim.ipynb 上做了简单的实验来验证这个结果。可能的解决方案有：
1. 使用 title 字段训练？ title 更容易做文本相似搜索？
2. 使用 word2vec embedding。小彬的经验是这样训练出来的模型可以做文本和文本间的相似搜索
3. 使用 *Learning Deep Structure-Preserving Image-Text Embeddings* 提到的 **Structure-preserving constraints**. 项目地址：https://dl.dropboxusercontent.com/u/17926179/embedding/embedding.htm
4. 使用 LSTM 得到文本向量


现在在跑着 4 方案，因为 4 方案最容易修改。训练完后执行 query_dumpy.py 和 server.py 就可以查看结果了。



#### jianzhu_VSE 代码流程

**数据特征抽取部分** (dump_data_pair.py): 抽取 jianzhu_tag.json 中的 detail, img_path, obj_id, url 信息。出于 Recall@K 计算较慢的考虑，验证集只划分 10000 个样本，其余样本做训练集。

**数据预处理部分** (pre_transforms.py): 将图片预处理成 2048 维的向量。文本使用 one-hot 编码。训练集得到图片向量 images_train 和图片描述 captions_train。验证集用于 Recall@K 的计算，需要去除重复的文本，并存储 caps_obj_id 和 imgs_obj_id 来判断图片和文本是否匹配。 caps_url, imgs_url, imgs_path 主要用于做补充信息的展示。测试集是训练集和验证集的合并，用于用户查询和寻找 good case。寻找 good case 时做模型效果验证（图片和文本是否匹配），所以也需要存储 caps_obj_id 和 imgs_obj_id。**预处理后需要对得到的文件做一些移动。**train 和 dev 数据全部放到 data 目录下相应位置，test 的 caps.txt 和 imgs.npy 放在 data 目录下相应位置，用来给 load_dataset 读取数据。 test 的 caps_url.json、imgs_url.json 和 imgs_path.json 放到 vse 目录下对应的 server 子目录中，用来做查询后的展示。<font color='red'>统一数据存放位置和变量命名。</font>

**模型训练部分** (train.py): 训练好的模型和对应的超参数会被保存下来。

**dump数据部分** (query_dump.py): 读取当前最好的训练模型和对应的超参数，将数据集中对应的图片和文本转换成图片向量和文本向量，并保存在 vse 目录下对应的 server 子目录中，以备查询之用。这里重新保存了训练模型和超参数，表示图片向量和文本向量是使用这个模型和超参数得到的。

**查询部分** (server.py)


#### 代码改造报告

数据特征抽取部分： 验证集和训练集的划分
数据预处理部分： word2vec 和 one-hot 的争论。(vocab.py ==> build_dictionary)
数据输入部分： 构造 batch (相同长度)
Loss 计算： 矩阵相乘的方法，一个正例，Batch_size-1个负例。Pytorch 如何自定义 Loss。
Recall@K 计算： 这部分代码目前已经改造完成。


使用 one-hot 编码，构造字典 (build_dictionary)。
不需要做参数初始化 (init_params, init_tparams)。
build_model 的时候，Loss 的定义方式。<font color='red'>Pytorch 如何自定义 Loss。</font>


也要参考 https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/model.py



#### pytorch l2norm 的计算

输入是 18x300 维的文本向量。表示有18个文本样例，每个文本向量的维度是300维。这里我们需要对每个文本向量做归一化。伪代码如下：
```Python
def l2norm(X):
    norm = torch.sqrt(torch.pow(X, 2).sum(1))
    X = torch.div(X, norm)
    return X
```
norm 的维度是 18x1，pytorch 除法不支持 broadcasting，只能通过 repeat 或者 expand 来间接实现。<font color='red'>repeat 做了真实的复制，而 expand 没有。所以 expand 不能用于需要修改的情况。</font>



#### Theano 计算图中间结果输出

建议先看 Theano tutorial [1](http://deeplearning.net/software/theano/tutorial/adding.html) 和 [2](http://deeplearning.net/software/theano/tutorial/examples.html)。

第一个例子基本能说明很多问题了。
```python
>>> import numpy
>>> import theano.tensor as T
>>> from theano import function
>>> x = T.dscalar('x')
>>> y = T.dscalar('y')
>>> z = x + y
>>> f = function([x, y], z)
>>> f(2, 3)
array(5.0)
```

第一步定义了两个 symbol x 和 y，这两个 symbol 都是 Variable。
第二步将 x 和 y 相加得到 z，z 也是一个 Variable。
第三步创建了一个函数将 [x, y] 作为输入，z 作为输出。
第四步传入真实值，得到运行结果。

在 VSE_theano 代码中，我们需要得到前传后的图片向量，查看它的 norm 。首先，定义图片特征输入 im 和经过前向传播得到的图片向量 images。
```Python
def build_images(tparams, options):
    """
    Computation graph for the model
    """
    opt_ret = dict()
    trng = RandomStreams(1234)

    # description string: #words x #samples
    im = tensor.matrix('im', dtype='float32')

    # Encode images (source)
    images = get_layer('ff')[1](tparams, im, options, prefix='ff_image', activ='linear')

    return trng, [im], images
```
调用 build_images 函数得到输入和输出。然后通过 theano.function 来定义计算节点 cal_images。
```Python
trng, im, images = build_images(tparams, model_options)
cal_images = theano.function(im, images, profile=False)
```
在得到真实的输入数据后，调用 cal_images 得到真实输出。
```Python
result = cal_images(im)
print 'result: ', numpy.linalg.norm(result[0])
```
最终得到的结果显示，<font color='red'>VSE 代码前向传播得到的图片向量并没有做归一化，这样使用点乘计算 Loss 是否正确？</font>






#### 训练集和验证集的划分
coco:  413915 / 5000
f30k:  145000 / 5070
f8k:  30000 / 5000

VSE代码将公开数据集的验证集只划分成5000左右，**主要是出于 Recall@K 计算上的考虑。**


#### conda 专题

theano程序要在gpu上运行的话需要pygpu的支持。有两种方式可以安装 pygpu：
- 如果使用 pip 安装 theano 的话，需要源码安装 pygpu，但中间有一步权限不够。
- 使用 conda 直接 conda install theano pygpu。

conda 安装: 下载 Anaconda_xxx.sh，`./Anaconda_xxx.sh`
conda 使用
- 在 ~/.zhsrc 中添加可执行路径
- [使用清华大学下载源](http://blog.csdn.net/huludan/article/details/52711550)
- conda 创建虚拟环境: `conda create --name env_name package`，conda 在虚拟环境创建的时候一定要指定安装一个或多个 package，这里随便填一个 numpy。虚拟环境的文件夹并不在当前目录下，而是统一放在 conda 的某个子目录下。
- 列出当前所有可用的环境及其路径: `conda info --envs`
- 进入和退出虚拟环境: `source activate env_name`、`source deactivate`
- 在虚拟环境中安装 package: `conda install package`
- 删除虚拟环境: `conda remove --name env_name --all`

出现的错误：
- undefined symbol: PyFPE_jbuf，原因是conda和主环境中的numpy冲突。[解决方案](https://stackoverflow.com/questions/36190757/numpy-undefined-symbol-pyfpe-jbuf): `pip uninstall numpy`
- kernprof command not found，解决方案: `conda install line_profiler`


使用GPU训练速度快了10倍左右，<font color='red'>目前的时间瓶颈在Recall@K。</font> 而 Recall@K 的瓶颈在 numpy.dot (17%) 和 numpy.argsort (77%) 。


#### arch Recall@K 的计算

arch 数据集没有公开数据集那么规范，每张图片都有5个对应描述，然后使用天然的 index 得到图片和文本是否匹配。Recall@K 只用到验证集的数据，对训练集没有影响。在作验证集数据处理的时候，读入的每个样例包含图文对以及它们所属的 obj_id。然后将所有文本做去重处理，得到 captions_dev。而与之对应的数组 caps_obj_id ，标识了每个 caption 对应的 obj_id。类似的，images_dev 也有 imgs_obj_id 来标识每个 image 对应的 obj_id。因此，在计算 Recall@K 时，我们通过图片和文本的 obj_id 来判断两者是否匹配。

i2t_arch() 计算以图搜文的 Recall@K。输入是全部的图片和全部文本向量。之前我们已经提到，在文本到文本向量的转换中，其顺序并没有改变。图片到图片向量的过程也是。遍历所有图片，依次和所有文本计算相似度。inds 按相似度排序的文本的序号。因为之前的处理中文本顺序并没有改变，所以我们可以通过序号直接找到其对应的 obj_id。caps_obj_id[inds] 得到按相似度排序的文本的 obj_id，numpy.where 得到与图片对应的 obj_id 在 caps_obj_id[inds] 中出现的位置。t2i_arch() 的过程也是类似的，这里不再赘述。


#### 阶段性总结

VSE 代码能够顺利跑起来了，代码的逻辑也基本看完了。在和黄超交流之后，把小批量数据按格式处理后放入 VSE 模型中，竟然成功运行了！！


#### VSE 代码学习

max_len_w：如果某句子包含的单词多于 max_len_w，则丢弃该样例（prepare_data func）。dispFreq 是 display frequence。


数据集f8k训练集、验证集和测试集分别包含30000,5000和5000个样例，每个样例包含captions和4096维图片特征向量（预训练VGG第19层输出，做了归一化）。

读取训练集和验证集的数据，并利用两者的 caption 构造字典。build_dictionary 返回的是 worddict 和 wordcount。两者都按照word出现的次数做了排序。worddict 是单词以及它们的 id。wordcount是单词以及它们出现的次数。

init_params 中，Word embedding 将 n_words 个单词分别映射成 dim_word 维的向量。Sentence encoder 将 dim_word 维的词向量 encode 成 dim 维的文本向量。Image encoder 将 dim_image 维的图片特征向量 encoder 成 dim 维的图片向量。init_tparams 将上面的 params 初始化成 theano 的共享变量。

程序将长度相同的句子放入同一个 batch 中，但 batch 长度不能超过 batch_size. x 是 captions 的集合，通过 worddict 将 captions 中的每个单词转换为对应的单词 id. x 是 (maxlen, n_samples) 维的向量。x_mask 应该是对应每个单词的权重。

build_model 函数中的 emb 将 x 从 id 转换为向量。如果 sentence encoder 是 bow 的话，直接将句子中的每个单词加权求和。mask[:, :, None] 将 mask 从二维向量变成三维向量，其中第三个维度是 1。 <font color='red'>gru 太复杂，这里先不看。</font>最后对句子向量做归一化。 图片特征向量经过 fc 层后得到图片向量。通过自定义的 contrastive_loss 来计算图片和文本的 Loss。

contrastive_loss 的输入是图片 (batch_size, dim) 和文本 (batch_size, dim)。将图片矩乘以文本矩阵得到相似度矩阵。相似度矩阵的对角线是图文对的相似度。每一行是图片和其它文本 (包括匹配文本)的相似度，每一列是文本和其它图片 (包括匹配图片)的相似度。有了这些之后，我们就可以计算 pairwise ranking loss 了!!!!

f_log_probs、f_cost 定义后没有使用。build_sentence_encoder 得到全部 sentence 的向量编码，build_image_encoder 得到全部 image 的向量编码。这两个向量编码用于后面 Recall@K 的计算。grads 通过 cost 计算出每个 tparams 的梯度，根据 grad_clip 做梯度截断。然后使用 optimizer 更新参数。

**计算图已经定义好了。后面的部分就是接入数据了。**

HomogeneousData 返回 Batch，且每个 Batch 的文本长度都相同。prepare() 统计每个长度下有多少句子以及每个句子的位置。reset() 对句子长度做乱序，对同一句子长度中的句子顺序做乱序。len_curr_counts 存储着每个长度下还有多少句子没有被使用。与之对应的是，len_indices_pos 存储着每个长度下访问到了哪个句子。

next() 如果当前句子长度是否还有句子没有被访问，那么跳出 while 循环去访问该句子长度下还未被访问的句子，然后跳到下一个句子长度 (注意这里不是把某个长度下的所有句子都访问完后，再访问下一个句子长度的句子)。否则，查看下一个句子长度是否还有句子未被访问，然后继续上面的操作。如果所有句子长度下的所有句子都被访问了。那么一个 epoch 就结束了，调用 reset 重置相关变量。

在访问某句子长度下还未被访问的句子时，首先通过 len_curr_counts 确定该长度下还有多少句子未被访问。然后和 batch_size 取一个较小值，并命名为 curr_batch_size。然后通过 len_indices_pos 得到当前长度访问到了哪个句子，从该句子开始访问 curr_batch_size 个句子，并通过 len_indices 得到这些句子的位置。更新 len_indices_pos 和 len_curr_counts。最后返回对应位置的句子和图片。

prepare_data 输入的是 batch，里面包含文本 caps 和图片特征 features。对 caps 做分词并通过worddict 转换为单词 id。抛弃到长度大于 maxlen 的句子。然后将文本向量和图片特征向量从 list 转成 numpy。最后弄个全是1的x_mask作为每个单词的权重。

encode_sentences 对验证集中的句子进行向量编码。ds 是一个可以按照长度访问句子的字典。[minibatch::numbatches] 的意思是从 minibatch 开始，每 numbatches 个取一个。然后将单词转换为单词 id，最后将沿着 f_senc ==> build_sentence_encoder 得到文本向量。在构造batch的时候句子的序号是按照相同长度被打乱的，但是到 features 的时候又根据句子的 id 进行重新排位，这时图片和文本又能够对应上了。

theano.function(input, output) output 中一般包含了怎么由 input 计算得到 output。

**Recall@K 的计算**

i2t() 计算以图搜文的 Recall@K 。输入是全部图片向量和全部文本向量。因为 f8k 数据集中每张图片都有5个对应的描述。在计算Recall@K的时候也利用到了这个特点，这是我们的数据中所不具有的。训练部分应该没有问题，但是 Recall@K 需要根据我们的数据集重新设计。 这里因为连续的5张图片都是一样的，所以只取第一张图片。然后和所有的文本向量计算相似度，得到相似文本的位置 inds。正确文本的位置是 5*index 到 5*index+5。这里依次查找每个正确文本出现在相似文本中的位置，取位置靠前的值作为模型的效果。

t2i() 计算以文搜图的 Recall@K 。输入同样是全部的图片向量和文本向量。因为连续的5张图片都是一样的，这里对图片做了5倍的压缩。这里图片原本是 500x300 维，做了压缩后变为 100x300 维。然后用连续相同的5个文本和图片计算相似度。这里 d 是 5x1000 维。表示5个文本和1000张图片的相似度。这里每个文本在向量编码上还是有差异的，所以对于每个文本都保留了 rank。其它的过程和 i2t() 基本类似，这里不再赘述。




#### 距离度量
在建筑项目中，需要在向量空间中计算某图片和文本的匹配程度。这里讨论两种计算方法 —— 欧式距离和余弦相似度。对于这两者的选择，思聪少爷说只能试，哪个好用哪个！

[余弦距离、欧氏距离和杰卡德相似性度量的对比分析](http://www.cnblogs.com/chaosimple/p/3160839.html)

余弦相似度是用向量空间中两个向量夹角的余弦值座位衡量两个个体间差异的大小的度量。如果两个向量的方向一致，即夹角接近零，那么这两个向量就相近。给定两个向量 $\mathbf{x}$ 和 $\mathbf{y}$，余弦相似度计算公式如下：
$$\text{sim}(\mathbf{x}, \mathbf{y}) = \cos \theta = \frac{\mathbf{x} \cdot \mathbf{y}}{||\mathbf{x}|| \cdot ||\mathbf{y}||}$$
其中分母表示两个向量的长度，分子表示两个向量的內积。

欧式距离和余弦距离各自有不同的计算方式和衡量特征，因此它们适用于不同的数据分析模型：
- 欧式距离能够体现个体树枝特征的绝对差异，所以更多的用于需要从维度的数值大小中体现差异的分析。
- 余弦距离更多的从方向上区分差异，而对绝对的数值不敏感。


[欧氏距离和余弦相似度的区别是什么？](https://www.zhihu.com/question/19640394) 归一化后，计算欧式距离，等价于计算余弦相似度。证明：两个向量 $\mathbf{x}$、$\mathbf{y}$，夹角为 $\theta$，经过归一化，他们的欧式距离 $D = (\mathbf{x}-\mathbf{y})^2 = \mathbf{x}^2 + \mathbf{y}^2 - 2||\mathbf{x}||\cdot||\mathbf{y}|| = 2-2 \cos \theta$

[漫谈：机器学习中距离和相似性度量方法](http://www.cnblogs.com/daniel-D/p/3244718.html) 余弦相似度会受到向量的平移影响。而皮尔逊相关系数具有平移不变性和尺度不变性。


word2vec 词向量都是经过归一化的，因此使用余弦距离和欧式距离是等价的。验证代码如下：
```Python
from gensim.models import Word2Vec
from numpy import linalg as LA

model = Word2Vec.load('static/word2vec-chi/word2vec_news.model')
for (key, value) in model.wv.vocab.iteritems():
    print LA.norm(model[key])

```

**关于距离度量方法的总结如下：**

- 使用 L2 距离度量方法。优点是 PyTorch 实现了 TripletMarginLoss，缺点是 Recall@K 的计算比较慢。可以考虑改进 L2 的计算方法或者改用 sklearn 的 NearestNeighbor。
- 使用向量(norm)內积方法。优点是 Recall@K 的计算速度比 sklearn NearestNeighbor 都要快3倍，同时 visual semantic embedding 的官方代码也是这么实现的。缺点是 Loss 函数需要自己实现。




#### Theano 相关
**TypeError**: shared variable(float32) doesn't has the same type with update variable(float64). Theano 默认的 floatX 是 float64，但它也提供了[接口](http://deeplearning.net/software/theano/library/config.html)，将其改成 float32. 有两种修改方式：
1. 在程序运行时加入配置变量： `THEANO_FLAGS='floatX=float32,device=cuda0' python <myscript>.py`. 但我们的程序跑在 ipython 中，好像用不了这种方式。
2. 修改 Theano 配置文件 ~/.theanorc，虽然我们 Theano 安装在虚拟环境中，但是配置文件还是在 ~/.theanorc，如果没有的话如要自己创建。然后在文件中加入下面代码：
```
[global]
floatX = float32
```

Theano 使用 GPU 十分简单，不用做变量上的改动，只需要修改配置文件，例如指定使用第0块gpu：
```
[global]
device = gpu0
floatX = float32
```

#### 距离度量在代码流程中需要保持一致
在训练阶段，使用的是 TripletMarginLoss，里面使用的是 L2 距离度量。在 Recall@K 计算的时候，使用的是 np.dot 內积距离度量。**这样的做法是不对的**。这里通过查看 TripletMarginLoss 的源码，在 Recall@K 计算的时候使用了 L2 距离度量。



#### Recall@K 的计算
这里尝试了两种方案。第一种方案使用 sklearn 的 NearestNeighbors，第二种方案参考 [ryankiros/visual-semantic-embedding](https://github.com/ryankiros/visual-semantic-embedding/blob/master/evaluation.py)，使用numpy.dot + numpy.argsort。以下是两个方案的运行结果。

第一次实验，numpy 放在 sklearn 前面，得到的结果是：
- numpy: img\_top\_k(94/15933)、text\_top\_k(440/54552)、**210s**
- sklearn: img\_top\_k(158/15933)、text\_top\_k(488/54552)、**721s**

第二次实验，numpy 放在 sklearn 后面，得到的结果是：
- numpy: img\_top\_k(102/15933)、text\_top\_k(431/54552)、**238s**
- sklearn: img\_top\_k(184/15933)、text\_top\_k(447/54552)、**751s**


可以看到，numpy 方案比 sklearn 方案的计算速度快很多，但计算结果是否正确还有待进一步分析！ (內积对于相似度而言没有任何意义。需要对图片和文本都做归一化，得到的余弦相似度才是有意义的。)
numpy 方案的瓶颈主要在 numpy.dot 和 numpy.argsort 中，没有办法做提升。


#### Pymongo
参照论文 Show, Attention and Tell: Neural Image Caption Generation with Visual Attention 的 Attention 实现，需要把图片表示从原来的 512 维提高到 7x7x512 维，存储空间几乎增加了30倍。于是折腾了数据压缩方案 h5py 和 数据库方案 pymongo。

Pymongo 每个文档最大为 16 Mb，而有些建筑项目的图片有 50+ 张，保存成 7x7x512 的图片特征会突破文档存储容量的上限。解决方法之一是使用 [GridFS](https://api.mongodb.com/python/current/examples/gridfs.html)。这里使用的是比较工程的方法，如果项目图片超过 40 张，则丢弃 40 张后的图片。



#### 强大的 jieba 分词
- `jieba.analyse.set_stop_words(filename)` 自定义关键词提取时所使用停止词。
- `jieba.analyse.extract_tags(sentence, topK, withWeight, allowPOS)`提取关键词，参数包括待提取文本，返回几个 TF/IDF 权重最大的关键词，是否一并返回关键词权重，筛选指定词性的词。


#### PyTorch RNN 使用指南
官方文档地址： http://pytorch.org/docs/nn.html#rnn 

**(1) 声明 RNN 对象**
传入的参数为文档中的 Parameters，依次为 input_size、hidden_size、 num_layers ...
```Python

input_size, hidden_size, num_layers = 300, 300, 1
rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
```
这里声明了一个 RNN 对象并命名为 rnn。这个 RNN 的 input_size 为 300，hidden_size 为 300，num_layers 为 1。batch_first 为 True，表示 rnn 前向传播时输入和输出的格式为 (batch, seq, feature)。

**(2) 对于变长输入的处理**
 PyTorch 提供了 `torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)` 来处理变长输入。原理是将 Batch 中所有样例的长度设置成 Batch 中样例长度的最大值，使用 lengths 来得到 padded 的位置。对于这些位置，RNN 是不需要前向传播，也不需要计算 loss 和更新梯度的。
 
 input 要求按长度降序排序。如果 batch_first 为 True，input 的格式为 `BxTx*`，B为 batch_size，T 为样例长度的最大值，* 为特征数。下面**通过一个例子来做详细地讲解**。
 
 - 在这个例子中，我们构造了 5 个长度不一的句子，句子中的每个单词使用 input_size 维的向量进行编码。
```Python
import numpy as np

s1 = np.random.randn(3, input_size)
s2 = np.random.randn(7, input_size)
s3 = np.random.randn(18, input_size)
s4 = np.random.randn(9, input_size)
s5 = np.random.randn(27, input_size)

s = [s1, s2, s3, s4, s5]

batch_size = len(s)
``` 
- 按照函数的要求，对句子按照长度进行排序，并得到函数的参数 `lengths`

```Python
s_sort = sorted(s, key=lambda sen:sen.shape[0], reverse=True)
lengths = [sen.shape[0] for sen in s_sort]
```
- 构造`input`的过程比较复杂。首先需要得到一个全零的 (batch_size, max_len, input_size) 的数组，然后使用切片技术将排序好的句子依次填入数组的相应位置。

```Python
max_len = s_sort[0].shape[0]
sen_input = np.zeros((batch_size, max_len, input_size), dtype='float32')

for i in range(batch_size):
    sen_input[i, :lengths[i], :] = s_sort[i]
    
sen_input = Variable(torch.from_numpy(sen_input))

sen_input_padded = torch.nn.utils.rnn.pack_padded_sequence(sen_input, lengths, batch_first=True)

```
这里需要注意的是，如果 np.zeros  声明时不加 dtype='float32'，得到的是 float64 的数，而 float64 的数组会被 torch.from_numpy 转换成 DoubleTensor，导致后面的数据类型错误。这里也可以通过 Tensor.float() 将 DoubleTensor 转换成 FloatTensor。

可以通过 arr.dtype 查看 numpy 数组的类型。可以使用 arr.astype(dtype) 将 numpy 数组  arr 的类型转换成 dtype. 

**(3) rnn 的前向传播**
```Python
output, hn = rnn(sen_input_padded)
```

rnn.forward(input, h_0)，input 的参数为 (seq_len, batch, input_size)，h_0 的参数为 (num_layers\*num_directions, batch, hidden_size)。官方文档的例子中给出了 h_0 的显式使用方法，在图文互搜中，我们可以将图片特征编码作为 h_0 输入到 rnn 中。如果 h_0 没有给出，推测程序会根据声明 RNN 对象时的参数 hidden_size, num_layers, bidirectional 得到 num_layer\*num_directions 和 hidden_size，根据 batch_first 和 input 得到 batch，然后通过某种方法随机初始化得到 h_0



#### Python 排序
Python 的排序函数有两种，`list.sort()`和`sorted(list)`。sorted(list) 返回排序后的 list，不会对原始 list 做改动。list.sort() 直接对原始 list 进行修改排序。

**sorted(list) 的用法如下：**


- 对由 tuple 组成的 List 排序
```Python
>>> students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10),]
>>> from operator import itemgetter, attrgetter  
>>> sorted(students, key=itemgetter(2))   # sort by age
('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]  
```

- 对字典排序
```Python
>>> d = {'data1':3, 'data2':1, 'data3':2, 'data4':4} 
>>> from operator import itemgetter, attrgetter   
>>> sorted(d.iteritems(), key=itemgetter(1), reverse=True)  
[('data4', 4), ('data1', 3), ('data3', 2), ('data2', 1)]
``` 





#### Image Caption 的几份 Github

- https://github.com/ruotianluo/neuraltalk2.pytorch
- https://github.com/eladhoffer/captionGen
- https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/09%20-%20Image%20Captioning



### 第一个收敛的版本

#### 1. 数据集的问题 
在处理 archgo 数据集的时候：

- 去掉包含 “梯” 的项目后，剩下 3655 个项目，train_loss (Adam) 一直在 0.5 附近震荡。 
- 如果不去掉，有 5275 个项目（多了 1620 个项目），train_loss (Adam) 收敛到 0.34 。

为了验证是否因为数据集大小导致的效果差异，我只取带梯数据集的前 3655 个项目，train_loss (Adam) 收敛到 0.35 。

这说明并不是因为数据集大小导致效果的差异。目前认为可能的原因是 **类别不均衡、负采样问题**，我统计了每个项目的第一个标签，排在第一的是【一梯二户】，有 1152 个； 排在第二的是【商业综合体】，有 313 个； 两者之间的差距是巨大的。我甚至怀疑收敛的原因是模型把大量的项目预测成梯户，这样也能获得不错的效果。关于这一点，后面还需要做实验来验证这个假设。目前的想法是把每个类别的准确率输出。


#### 2. 梯度下降方法的问题

<font color=red>梯度下降方面的内容比较多，暂时放在后面慢慢细究</font>。相关内容可以参考：

- An overview of gradient descent optimization algorithms
- AI 带路党 —— 比Momentum更快：揭开Nesterov Accelerated Gradient的真面目 - 知乎专栏
- 路遥知马力 —— Momentum - 无痛的机器学习 - 知乎专栏
- http://cs231n.github.io/neural-networks-3/



带梯数据集，5275 个项目，收敛效果：Adam > SGD+动量 > SGD 。值得一提的是，SGD 也是能够缓慢收敛的，迭代 21 次后 train_loss 降到 0.41 。也就是说模型不收敛主要是数据的问题，收敛地好不好是梯度下降算法的问题。



#### 3. 深度学习调参技巧

- https://zhuanlan.zhihu.com/p/24720954
- https://www.zhihu.com/question/25097993
- http://dataunion.org/26881.html
- http://blog.csdn.net/zr459927180/article/details/51577055
- http://www.9u99.com/37760/
- https://cethik.vip/2016/10/02/deepnn3/


#### 4. Pytorch eval()
http://pytorch.org/docs/nn.html?highlight=eval#torch.nn.Module.eval
Sets the module in evaluation mode. **This has any effect only on modules such as Dropout or BatchNorm**.

Pretrained cnn model Resnet 中包含 BatchNorm layer，在 no pre_transforms 模型中，传入 Resnet 的 Batch 来自于 shuffle 后的不同项目的图片，这个 Batch 和训练数据集更接近独立同分布，也就是说每个 Batch 的均值和方差是差不多的，所以 no pre_tranforms 模型训练初期不会震荡。对于 pr_transforms 模型来说，传入 Resnet 的 Batch 是同个项目下的所有图片，不同项目图片的均值和方差的差异是十分巨大的，导致 pre_transforms 模型在训练初期就震荡。 解决方法是 `resnet = resnet.eval()`






### python dict 按值排序
```
import operator

dict = {'dog': 20, 'cat': 8, 'panda': '30', 'banana': 17}
dict = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

for (key, value) in dict:
	print key, value
```


### json dump 和 dumps
为了在 debug 的时候弄清楚数据处理的每个步骤有没有问题，特别是送进模型里面的每个 Batch 有没有问题（负例），决定实现 processed data_visualization 。想法是记录 data_util 中选中的图片以及使用的句子负例。但是因为在 data_util 中做了数据的 shuffle，记录的过程会更加棘手。因此，经过分析之后，将**数据 shuffle 的过程放在 data_pair_archgo_tag.ipynb 中进行处理**。

原先使用 json dumps 序列化数据，实例代码如下：

```Python

# json.dumps
with open('xxx.json', 'w') as f_write:
	for k, item in enumerate(all_data):
		# get sentence, images from item
		f_write.write(json.dumps(sentence, images)+'\n')

# json.loads
with open('xxx.json', 'r') as f_read:
	data_pair = map(json.loads, f_read.readlines())
	(sentence, images) = data_pair[k]
```
json.dumps 一个个地写入 (sentence, images) pair，在我们的需求中，需要收集所有的 (sentence, images) pair，shuffle 之后在一次性写入到文件中。因此这里使用的是 json.dump，这里需要同时注意 load 上的差异

```Python

# json.dump
extract_data = []
for k, item in enumerate(all_data):
	# get sentence, images from item
	extract_data.append((sentence, images)) 
	
json.dump(extract_data, open('xxx.json', 'w'))


# json.load
data_pair = json.load('xxx.json', 'r')
(sentence, images) = data_pair[k]
```




### Python multiprocess 
多进程数据处理是个伪命题，原来的数据处理程序也是跑满全部 cpu 的。从实际情况上看也是这样。500 个 items，data_processed_multiprocess.ipynb 开全部 cpu (12块) 的运行时间是 267.74s，data_processed.ipynb 的运行时间是 263s 。



### PyCharm ipynb 同步问题
其实所有的代码，包括 python 代码和 ipynb 代码，都应该在服务器上修改并运行。但是服务器并不支持 PyCharm IDE，而 PyCharm 又有许多优越的特性。

所以对于 Python 代码来说，现在采取的模式是在本地修改，自动同步到远程服务器，并在服务器上运行。

对于 ipynb 代码来说，如果是在本地修改并运行，那么在一次完整性的修改完成后，需要手动同步到服务器。方法是在 PyCham 中选中修改的 ipynb 文件、 
Deployment、Upload to server； 在服务器修改的情况类似，只是最后一步变为 Download from server.

### Shuffle 的意义
**Shuffle 的目的是使训练集和验证集同分布**。因此，只需要在读取数据的时候做一次 shuffle，然后将数据分为训练集和验证集。在以后的每个 epoch 中都不再需要对训练集或验证集单独做 shuffle.


### Dive into plt
画图的一般步骤是：定义一个画布，在画布上添加内容，显示画布。

plt.figure(arg) 的功能为定义一个画布，arg 可以设置画布的大小、背景颜色、线条宽度等等。没有使用 plt.figure(arg) 定义画布的话，系统会使用默认的设置来定义一张画布。

plt.imshow(arg) 的功能是往画布中添加内容。

plt.show() 的功能是将画布中的内容显示出来，然后刷新画布。如果没有定义的话，系统会在程序执行的最后自动添加 plt.show() 来显示画布内容。

plt 还设置到更多的内容，有时间的话再深入地理解。下面就代码中的具体问题进行分析：

```Python
plt.figure(figsize=(200,200))

for i in range(3):
	print sentence
	plt.imshow(Image.open(xxx).convert('RGB')
```
在这个样例中，因为没有显式调用 img.show()，在 for 循环的过程中，会不断地往画布中添加内容，并覆盖上一次画布中留下的内容。在 for 循环结束后，系统调用 plt.show() 在显示画布中最后留下的内容。当然，使用 plt.subplot() 的方式可以在 for 循环中依次将图片写入到不同的子图中，然后可以在最后显示出所有的图片。这时就出现了这种情况 —— 先显示所有的  sentence，然后显示所有的图片。

因此，我们在 for 循环中加入 plt.show()，每次添加内容后，显示内容并刷新画布。代码如下：
```
plt.figure(figsize=(200,200))

for i in range(3):
	print sentence
	plt.imshow(Image.open(xxx).convert('RGB')
	plt.show()
```
plt.show() 显示内容并刷新画布。在第一次循环中，使用的是 for 循环外定义的画布，figsize 是 (200, 200)，然后在循环的末尾刷新画布。在第二次循环中，上一次的画布被刷新了，但新画布没有使用 plt.figure() 来显式生成。这时系统会使用默认的设置来定义一张画布，也就是说 figsize 不再是 (200, 200) 了。后面的循环也是类似的过程。因此，为了让画布设置在每个画布中都有效，我们在循环的内部定义画布，同时去掉默认的轴坐标显示：
```Python
for i in range(3):
	plt.figure()
	plt.axis('off')
	print sentence
	plt.imshow(Image.open(xxx).convert('RGB')
	plt.show()
```
在上面的代码中，如果去掉 `plt.show()` 的话，每个循环会生成一张画布并填充内容，同时显示 sentence。最后看起来的效果就是先显示所有的 sentence，在循环结束后显示所有的画布。





### plt title 中显示中文
使用 plt 显示图片时，先显示所有的文字，然后再显示所有的图片。而我们想要的是图片和文字交替显示。下面给出的解决方案是将文字放在 plt title 中显示。

首先使用下面这段代码显示系统可用的中文字体
```Python
#! /usr/bin/env python
# -*- coding: utf-8 -*-
from matplotlib.font_manager import FontManager
import subprocess

fm = FontManager()
mat_fonts = set(f.name for f in fm.ttflist)

output = subprocess.check_output(
    'fc-list :lang=zh -f "%{family}\n"', shell=True)
# print '*' * 10, '系统可用的中文字体', '*' * 10
# print output
zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
available = mat_fonts & zh_fonts

print '*' * 10, '可用的字体', '*' * 10
for f in available:
    print f
```
然后在代码中进行参数设置
```Python
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
```
`Droid Sans Fallback` 是上面那段代码输出的可用中文字体。








### data_processed 后续代码接入
data_processed 部分新加了两份代码 —— datautil_processed.py 和 main_processed.py 。和之前的不同之处在于，为了节省运行时间，ArchtectDataset 只初始化一次，也就是说只读取一次 json 数据，后面每个 batch 并不再次读取数据，而是对数据集做 shuffle.




### Python 迭代器
使用 for ... in 或者 enumerate 遍历时，如果遇到越界错误，会直接停止遍历，而且不会报错。一般来数，数据越界会报 IndexError 错误，这里应该是被 iterator 捕获并当做 StopIterator 来处理了。 
```Python
class IteratorBug(object):
    def __init__(self):
        self.arr = [1, 2, 3, 4, 5, 6]
    def __getitem__(self, index):
        if index == 3:
            # Index out of range
            # But do not raise error when iterate
            return self.arr[11]
        else:
            return self.arr[index]

if __name__ == '__main__':
    it = IteratorBug()
    for item in it:
        print item
```



### PIL 序列化

json 不能序列化 PIL image，但是 cPickle / pickle 可以，因为它们是专门的 Python 序列化工具。cPickle 比 pickle 快，缺点是 cPickle 不能继承。 使用方法：

```
try:
   import cPickle as pickle
except:
   import pickle
   
pickle.dump(dumps_pair, open(root_dir+'data_processed.pickle', 'wb'))
data_pair = pickle.load(open(root_dir+'data_processed.pickle'))
```

pickle 和 json 虽然 API 都差不多，但是使用 f_write(pickle.dumps((sentence, images))+'\n') 每次写入一个 item，然后使用 map(pickle.loads, f_read.readlines())时，会报 Pickle data was truncated。f_read.readline() 打印出来后发现，第一行只是第一个 item 的一部分。

PIL 序列化占用的内存太大了。假设一张图片是 20k，使用 PIL 打开后会增大 10 倍（图片解压缩），假设每个项目有20张图片，所以保存一个项目的 PIL 图片就需要 20k * 10 * 20 = 40 M 的存储空间。所以**图片还是要预处理到图片特征这个层面**。



### 数据处理加快程序运行
小彬师兄提到数据的读取占用了程序运行的大部分时间，可以考虑把这部分内容提前处理好。我思考了一下，使用 PIL.Open() 打开图片，使用 img_transforms 对图片做预处理，然后使用 resnet 得到图片的特征向量； 使用 sen_transforms 对文本做预处理； 这些都是可以预先处理好的内容。

**使用 line_profiler 模块来查看某个函数执行每行代码所需占用的 CPU 时间**。 使用方法

- `$ pip install line_profiler` 安装 line_profiler 模块

- 在需要检测运行时间的函数前面加上 @profile

- `$ CUDA_VISIBLE_DEVICES=2 kernprof -l -v main.py` 得到结果

发现上面提到的数据处理占用了 90% 以上的时间，于是写了 data_processed.ipynb 来对数据进行预处理。值得一提的是：

- 使用 `resnet.fc = torch.nn.Dropout(p=0)` 覆盖原来的 resnet.fc，直接将 fc 层的输入作为输出，因为 fc 层的输入就已经是我们要的图片的特征了。

- json.dumps() 不能序列化 tensor，需要通过 `tensor.numpy().tolist()` 将其转换为 list 后才能进行序列化。


### Fix Image size 0
有些图片 size 为 0，使用 PIL.open 打开图片时会报错，这里在 data_pair_archgo_tag.ipynb 中使用 os.path.getsize() 去除掉这些图片。



### CossineEmbeddingLoss() + cuda(num)

CosineEmbeddingLoss() + cuda(num) 出现问题， https://github.com/pytorch/pytorch/issues/1192， PyTorch 的版本号是 0.1.10，运行环境是95和96服务器，num 设置成 1 或 2 时会报`arguments are located on different GPUs` 错误。 num设置成 0 或 不设置的时候不会报错。

解决方案：

- 小彬师兄给出的解决方案是： xx.py 代码中使用 cuda() 替代 cuda(num)，这时候代码运行不会报错，但是默认是在 GPU 0 上运行程序。然后使用 `CUDA_VISIBLE_DEVICES=num python xx.py` 限制程序在 GPU num 上运行。


- soumith 给出的解决方案是：升级 PyTorch 为 0.1.11，and this works fine for me。

