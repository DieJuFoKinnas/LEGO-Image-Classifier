
# Classifying Images of LEGO pieces

This project aims to build a simple CNN classifier for lego pieces, which will be trained on synthetic data.

### Synthetic data

Since synthetic data is basically free for this use case, it should be leveraged and also mixed with real data for additional performance([this paper](https://arxiv.org/abs/1706.06782) shows that for their use case mixing caused a huge increase in mean average precision(see figure 9)). After the network has been trained on synthetic data it would be easy for the network to display a ranking list of pieces it believes match. This would make it easy for a human to find the correct label in that ranking and correct potential mistakes. Unfortunately, to mix real with synthetised data, we would have to actually take pictures of real lego pieces...

#### Software for synthetic data

First you will need to install blender and the ldraw import [plug-in](https://github.com/TobyLobster/ImportLDraw) and install it. Then you can run scripts by changing to script view and selecting the script you want to run(or modify and run). Note that on OSX and on Linux you need to run blender from the console to see debugging information that the executed script prints. Make sure you run blender from the project directory, because the script is using relative paths and remember to set the correct ldraw parts path inside the script.

### Model Architecture

Capsule networks seem to be the perfect match for this use case, but training them takes many times the time to train a regular convolutional neural network. Thus it is probably wiser to not use them, if it is only to gain a few percentage points in performance.

Initially, the high number of classes seemed to be a major issue, but thanks to [candidate sampling and noise contrastive estimation(NCE)](https://www.tensorflow.org/extras/candidate_sampling.pdf) we can approach the task like a regular classification task. NCE doesn't compute probabilities for *every single* class, but takes the *true class label* and mixes it with a few *fake* ones, so the network has to figure out which of the class labels corresponds to the image. Note that the use of this loss was inspired by [word2vec](https://www.tensorflow.org/tutorials/word2vec), where the problem is similar, since there are millions of word classes.

Luckily tensorflow has a preimplemented nce-loss-function, so it was easy to construct an example of its usage for MNIST.

The neural network system could also be enhanced by a weighting scale, because one can immediately discard a large amount of pieces if they have a very different weight.


### Classifying barely seen classes

I think it might be possible to classify piece classes that weren't even present in the training set by giving 1 example of that class... We will see.
