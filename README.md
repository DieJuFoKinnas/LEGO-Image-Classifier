
# LEGO Piece Image Classification

The final goal is to build a successfull classifier that takes in an image from some angle of a lego piece and then let a neural(or capsule) network determine which one it is.


## Model Architecture

a) A capsule net that maps images into a latent space and a 3d convolutional net that maps the voxel representation of a 3d model of a piece into the same space. On that space you compute a metric like the cosine similarity to find the model a image is closest to.

b) Only one Capsule net that maps images into a latent space. Then the true label of the image is estimated like this: Let a cluster of images be a set of images which belong to a piece. Then a image would belong to a cluster if the distance to it(in the latent space) is the lowest.

a) has fast computation, while b) can be slow to make predictions and needs carefull optimization. But b) has less parameters and requires no 3d model and with that also no generation of a voxel representation of the model.


## Synthetic data

Since synthetic data is basically free for this usecase, it should be leveraged and also mixed with real data for additional performance(https://arxiv.org/abs/1706.06782 shows that for their usecase the mixing caused a huge increase in mean average precision(see figure 9)). After the network has been trained on synthetic data it would be easy for the network to display a rank of matching pieces. This would make it easy for a human to find the correct label in that rank and correct potential mistakes.

### Generating data in blender

First you will need to install blender and the ldraw import [plug-in](https://github.com/TobyLobster/ImportLDraw) and install it. Then you can run scripts by changing to script view and selecting the script you want to run(or modify and run). Note that on OSX and on Linux you need to run blender from the console to see debugging information that the executed script prints. On Windows you can open a console window inside blender.
