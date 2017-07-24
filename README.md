
### 如何运行代码？


##### 1. 安装环境

```Python
pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
pip install torchvision
pip install gensim
pip install hyperboard
```


##### 2. 运行项目代码


```Python
python dump_data_pair.py
python pre_transforms.py
hyperboard-run --port 5020
python test.py
python query_dumpy.py
python server.py

```



### 对代码的详细解释

`dump_data_pair.py` 从原始数据 jianzhu_tag.json 中抽取特征 title, detail, image, (url)，同时对数据做 shuffle.

`pre_transforms.py` 将图片预处理成 2048 维的向量。文本使用 one-hot 编码。训练集得到图片向量 images_train 和图片描述 captions_train。验证集用于 Recall@K 的计算，需要去除重复的文本，并存储 caps_obj_id 和 imgs_obj_id 来判断图片和文本是否匹配。 caps_url, imgs_url, imgs_path 主要用于做补充信息的展示。测试集是训练集和验证集的合并，用于用户查询和寻找 good case。寻找 good case 时做模型效果验证（图片和文本是否匹配），所以也需要存储 caps_obj_id 和 imgs_obj_id。**预处理后需要对得到的文件做一些移动。**train 和 dev 数据全部放到 data 目录下相应位置，test 的 caps.txt 和 imgs.npy 放在 data 目录下相应位置，用来给 load_dataset 读取数据。 test 的 caps_url.json、imgs_url.json 和 imgs_path.json 放到 vse 目录下对应的 server 子目录中，用来做查询后的展示。

`test.py` 训练好的模型和对应的超参数会被保存下来。这一步包含数据的读取和词典构造、处理不同长度的句子、计算 pairwise ranking loss、计算 Recall@K 等，这些会在后面的文档中详细进行说明。

`query_dump.py` 读取当前最好的训练模型和对应的超参数，将数据集中对应的图片和文本转换成图片向量和文本向量，并保存在 vse 目录下对应的 server 子目录中，以备查询之用。这里重新保存了训练模型和超参数，表示图片向量和文本向量是使用这个模型和超参数得到的。

`server.py` 搭建图文互搜网站，供用户输入建筑描述或建筑图片，返回相应的查询结果


这里我们重点关注 train.py，这份代码是图文互搜项目的核心代码。

##### 1. 处理不同长度的句子

读取训练集和验证集的数据，并利用两者的 caption 构造字典。build_dictionary 返回的是 worddict 和 wordcount。两者都按照word出现的次数做了排序。worddict 是单词以及它们的 id。wordcount是单词以及它们出现的次数。


HomogeneousData 返回 Batch，**且每个 Batch 的文本长度都相同**。`prepare()` 统计每个长度下有多少句子以及每个句子的位置。`reset()` 对句子长度做乱序，对同一句子长度中的句子顺序做乱序。len_curr_counts 存储着每个长度下还有多少句子没有被使用。与之对应的是，len_indices_pos 存储着每个长度下访问到了哪个句子。

`next()` 如果当前句子长度是否还有句子没有被访问，那么跳出 while 循环去访问该句子长度下还未被访问的句子，然后跳到下一个句子长度 (注意这里不是把某个长度下的所有句子都访问完后，再访问下一个句子长度的句子)。否则，查看下一个句子长度是否还有句子未被访问，然后继续上面的操作。如果所有句子长度下的所有句子都被访问了。那么一个 epoch 就结束了，调用 reset 重置相关变量。

在访问某句子长度下还未被访问的句子时，首先通过 len_curr_counts 确定该长度下还有多少句子未被访问。然后和 batch_size 取一个较小值，并命名为 curr_batch_size。然后通过 len_indices_pos 得到当前长度访问到了哪个句子，从该句子开始访问 curr_batch_size 个句子，并通过 len_indices 得到这些句子的位置。更新 len_indices_pos 和 len_curr_counts。最后返回对应位置的句子和图片。

`prepare_data()` 输入的是 batch，里面包含文本 caps 和图片特征 features。对 caps 做分词并通过worddict 转换为单词 id。抛弃到长度大于 maxlen 的句子。然后将文本向量和图片特征向量从 list 转成 numpy。

`encode_sentences()` 对验证集中的句子进行向量编码。ds 是一个可以按照长度访问句子的字典。[minibatch::numbatches] 的意思是从 minibatch 开始，每 numbatches 个取一个。然后将单词转换为单词 id，最后将沿着 f_senc ==> build_sentence_encoder 得到文本向量。在构造batch的时候句子的序号是按照相同长度被打乱的，但是到 features 的时候又根据句子的 id 进行重新排位，这时图片和文本又能够对应上了。


##### 2. PairwiseRankingLoss 的计算

PairWiseRanking 的输入是图片 (batch_size, dim) 和文本 (batch_size, dim)。将图片矩乘以文本矩阵得到相似度矩阵。相似度矩阵的对角线是图文对的相似度。每一行是图片和其它文本 (包括匹配文本)的相似度，每一列是文本和其它图片 (包括匹配图片)的相似度。有了这些之后，我们就可以计算 pairwise ranking loss 了!!!!

#### 3. Recall@K 的计算

arch 数据集没有公开数据集那么规范，每张图片都有5个对应描述，然后使用天然的 index 得到图片和文本是否匹配。Recall@K 只用到验证集的数据，对训练集没有影响。在作验证集数据处理的时候，读入的每个样例包含图文对以及它们所属的 obj_id。然后将所有文本做去重处理，得到 captions_dev。而与之对应的数组 caps_obj_id ，标识了每个 caption 对应的 obj_id。类似的，images_dev 也有 imgs_obj_id 来标识每个 image 对应的 obj_id。因此，在计算 Recall@K 时，我们通过图片和文本的 obj_id 来判断两者是否匹配。

i2t_arch() 计算以图搜文的 Recall@K。输入是全部的图片和全部文本向量。之前我们已经提到，在文本到文本向量的转换中，其顺序并没有改变。图片到图片向量的过程也是。遍历所有图片，依次和所有文本计算相似度。inds 按相似度排序的文本的序号。因为之前的处理中文本顺序并没有改变，所以我们可以通过序号直接找到其对应的 obj_id。caps_obj_id[inds] 得到按相似度排序的文本的 obj_id，numpy.where 得到与图片对应的 obj_id 在 caps_obj_id[inds] 中出现的位置。t2i_arch() 的过程也是类似的，这里不再赘述。

