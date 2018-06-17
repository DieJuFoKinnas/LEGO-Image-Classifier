
# LEGO Robotic Arm Classifier

The final goal is to build some kind of robot arm that is capable of separating LEGO pieces from a pile and then identify which *exact piece* it is, so it can sort it accordingly. Then it would be possible to search a database for which LEGO sets can be built with it and also generate a list of which pieces a human must retrieve from each bin to have all the pieces one needs for the set. This solves the problem of entropy increasing in one's LEGO collection, and for big LEGO fans it does save a huge amount of time that would be spent on sorting the collection and mantaining it sorted.

##### Why a robotic arm as opposed to a system of conveyor belts?

* Robotic arms are cool
* One can easily repurpose a robotic arm for other tasks/experiments
* It does not take up a lot of space
* One can change the layout of the sorting bins as one pleases, as long as the robotic arm is reconfigured
* Templates for 3D printing robotic arms can be found online, so one doesn't have to spent a lot of time engineering the intricacies of a conveyor belt system

### Piece separation

It remains to be tested whether or not one can setup the robot arm to blindly try to grab a piece out of the container. Maybe it would only be able to grab a piece 50% of the times, which would look a bit silly, though it should be okay for the first attempt.

If this does not work at all(or in ~10% of the cases), the complexity of the project would be considerably increased.

* One solution would require us to either build a physical machine to separate the pieces, so one could use a kind of shovel on the arm's tip to pick the pieces up easily

* One could also build a neural network to use multiple cameras to predict the exact position and rotation the of the "robotic hand" on the arm's tip, such that when the "hand" closes it grabs a piece. That neural network would most likely need to use some kind of attention mechanism, because higher definition image data is only required for the region, where the piece of interest lies.

### Synthetic data

Since synthetic data is basically free for this use case, it should be leveraged and also mixed with real data for additional performance([this paper](https://arxiv.org/abs/1706.06782) shows that for their use case mixing caused a huge increase in mean average precision(see figure 9)). After the network has been trained on synthetic data it would be easy for the network to display a ranking list of pieces it believes match. This would make it easy for a human to find the correct label in that ranking and correct potential mistakes.

#### Software for synthetic data

First you will need to install blender and the ldraw import [plug-in](https://github.com/TobyLobster/ImportLDraw) and install it. Then you can run scripts by changing to script view and selecting the script you want to run(or modify and run). Note that on OSX and on Linux you need to run blender from the console to see debugging information that the executed script prints. Make sure you run blender from the project directory, because the script is using relative paths and remember to set the correct ldraw parts path inside the script.

#### Data generation specifications

To know exactly how the data should be generated, the robotic arm setup should be finished, so one should be able to emulate the environment.

There a various ways to let the machine take pictures.
* One can let it move the LEGO piece in front of some kind of white(?) screen(still having it in its grip) and then let cameras take pictures from multiple angles, or just having the robotic arm move such that the camera has a different view of the piece. The "robotic hand" would partially obstruct the view on the piece, so classification accuracy could be decreased(though a smart neural network wouldn't mind). To use this with synthetic data one would need to somehow make a physics simulation with the gripper. The gripper can only grip the piece in a small amount of ways, so one could maybe take advantage of this with physics simulations.

* It might be wise to let it deposit the LEGO piece on a surface for a short amount of time, such that unobstructed pictures can be taken. But that introduces the seemingly trivial problem of picking the piece up again and requires an extra step. Nevertheless, this option would also increase the classification accuracy since a given piece has a small amount of ways it can lay on the ground, so not all rotations in 3d space are possible. To fully take advantage of this with synthetic data, one would have to determine the ways a piece can land on the ground, which would most likely require a physics simulation.

The first option saves a step and the neural network can most likely deal with a small obstruction of the piece, though it requires complicating the rendering process. We went with the first approach.


### Model Architecture

Capsule networks seem to be the perfect match for this use case, but training them takes many times the time to train a regular convolutional neural network. Thus it is probably wiser to not use them only to gain a few percents in performance.

Initially, the high number of classes seemed to be a major issue, but thanks to [candidate sampling and noise contrastive estimation(NCE)](https://www.tensorflow.org/extras/candidate_sampling.pdf) we can approach the task like a regular classification task. NCE doesn't compute probabilities for *every single* class, but takes the *true class label* and mixes it with a few *fake* ones, so the network has to figure out which of the class labels corresponds to the image. Note that the use of this loss was inspired by [word2vec](https://www.tensorflow.org/tutorials/word2vec), where they had a similar problem, since they had millions of word classes.

Luckily tensorflow has a preimplemented nce-loss-function, so it was easy to construct an example of its usage for MNIST.

The neural network system could also be enhanced by a weighting scale, because one can immediately discard a large amount of pieces if they have a very different weight.


### Classifying barely seen classes

I think it might be possible to classify piece classes that weren't even present in the training set by giving 1 example of that class... We will see.
