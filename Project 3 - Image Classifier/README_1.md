# Image-Classifier-Project

# Part 1
Udacity (Data Scientist Course)  - Create Your Own Image Classifier Project
Developing an Image Classifier with Deep Learning
In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. We'll provide some tips and guide you, but for the most part the code is left up to you. As you work through this project, please refer to the rubric for guidance towards a successful submission.

Remember that your code should be your own, please do not plagiarize (see here for more information).

This notebook will be required as part of the project submission. After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project. If you'd instead prefer to work on your local machine, you can find the files on GitHub here.

If you are using the workspace, be aware that saving large files can create issues with backing up your work. You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. If you're using VGGnets, the input to the dense classifier is something like 20000 units. Adding a layer of 1024 units (of 32-bit floating point values) after that leads to 20000 * 1024 * 32 \,\mathrm{bits} \approx 82 \, \mathrm{MB}20000∗1024∗32bits≈82MB just for that one weight matrix. In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. Keep an eye on the size of the checkpoint you create. You can open a terminal and enter ls -lh to see the sizes of the files. If your checkpoint is greater than 1 GB, reduce the size of your classifier network and resave the checkpoint.
