# Gensim训练词向量-维基百科语料库

### 语料库

#### 1、本次演示使用维基百科中文语料库中的下图语料。

维基百科中文语料库：[下载地址](https://dumps.wikimedia.org/zhwiki/)

![](https://upload-images.jianshu.io/upload_images/19723859-e39d8b783caab344.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 2、下载后使用WikiExtractor从压缩包中提取正文文本。

WikiExtractor的github [下载地址](https://github.com/attardi/wikiextractor/blob/master/WikiExtractor.py)，复制代码将其命名为WikiExtractor.py，使用下面语句提取正文文本。

```
python3 WikiExtractor.py -b 500M -o output_filename input_filename.bz2
```

> WikiExtractor.py里面存放Wikipedia Extractor代码；
>
> -b 1000M表示的是以1000M为单位进行切分，有时候可能语料太大，我们可能需要切分成几个小的文件（默认），这里由于我需要处理的包只有198M，所以存入一个文件就行了，所以只需要设置的大小比198M大即可；
>
> output_filename：需要将提取的文件存放的路径；
>
> input_filename.bz2：需要进行提取的.bz2文件的路径；

#### 3、繁体字转换简体字

用文本编辑器打开wiki_00文件，可以看到提取出的语料中繁简混杂，所以我们需要借助工具将繁体部分也转换为简体。这里使用OpenCC工具化繁为简，可以通过下面的地址选择合适的版本，点击下载然后解压即可。

```text
opencc -i input_filename -o output_filename -c t2s.json
```

执行后的语料库文件样式：

![](https://upload-images.jianshu.io/upload_images/19723859-400a2e33b3da4e70.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 训练

对于word2vec来说，不论的Skip-Gram models还是CBOW models，他们的输入以及输出都是以单词为基本单位的，只是他们对应的输入以及输出不一样：

1. Skip-Gram models：输入为单个词，输出目标为多个上下文单词；
2. CBOW models：输入为多个上下文单词，输出目标为一个单词；

那么我们获取到的语料，必须要经过分词处理以后才能用于词向量的训练语料。

#### 1、分词

参考gensim_test.py中的segment方法

#### 2、训练

参考word2vec_model.py

#### 3、测试

参考gensim_test.py中的test方法

### 附录

#### 参考文章

[知乎-维基百科简体中文语料的提取](https://zhuanlan.zhihu.com/p/39960476)

[知乎-使用Gensim模块训练词向量](https://zhuanlan.zhihu.com/p/40016964)

#### 附件

由于训练好的model过大，我删掉了，可自行训练（本案例10分钟左右训练完毕）。下图是删掉的文件。

百度网盘：[下载链接](https://pan.baidu.com/s/1gXN3crasx_jMgI6oSxve-A) ，密码: gguk。

![](https://upload-images.jianshu.io/upload_images/19723859-3037e148c3786f45.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)