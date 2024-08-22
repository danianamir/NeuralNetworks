# Abstract

This report examines neural networks and the impact of various parameter values on their output using the EMNIST dataset, which consists of 47 classes. We explore the effects of parameters such as the number of layers, dropout techniques, and batch normalization on network performance.

# Introduction

To begin, we use a simple neural network and the PyTorch library for data classification. In neural networks, the number of layers varies depending on the selected dataset. Our goal in the first part is to investigate the impact of the number of layers on the network's accuracy by analyzing different layer counts and their corresponding outputs.

In the second part, we examine the dropout technique, where some connections between the nodes in the neural network are removed, transitioning the network from a fully connected state. Finally, we explore batch normalization, where input data is normalized.

# Methods

## Layer Depth

### Input Data

Initially, we stored the EMNIST dataset in a dataframe. This dataset consists of handwritten letters, with 112,000 samples for training and 18,000 samples for testing, totaling 47 different classes. Some sample images are shown below:

![Handwriting Sample 1](pic1.jpg)
*Figure 1: Handwriting Sample 1*

![Handwriting Sample 2](pic2.jpg)
*Figure 2: Handwriting Sample 2*

![Handwriting Sample 3](pic3.jpg)
*Figure 3: Handwriting Sample 3*
