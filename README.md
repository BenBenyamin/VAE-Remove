# MSAI 495 - Generative AI - Project 1 | Image Generation
## Ben Benyamin

### Overview

In this project, the objective was to remove objects from images—rather than generate them—using techniques covered in class. By leveraging Variational Autoencoders (VAEs), the goal was to learn the average latent representation of each object class. This representation captures the "essence" of the class, which can then be subtracted from the image’s latent space to effectively remove the object from the reconstructed output.

The dataset used for this task was [STL-10](https://cs.stanford.edu/~acoates/stl10/), which consists of 10 object classes commonly used in unsupervised and representation learning tasks. [Stanford Cars](https://huggingface.co/datasets/tanganke/stanford_cars) was also evaluated, though its training process are omitted here for brevity.
Please see the notebook for this project here --- [Notebook](train/Project1.ipynb)
