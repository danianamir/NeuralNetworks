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




### Neural Network

To start working with the data, we built a neural network with 5 layers:
- Input: 768 inputs corresponding to the number of pixels in each image.
- Layer 1: 256 units
- Layer 2: 128 units
- Layer 3: 64 units
- Layer 4: 32 units
- Layer 5 (Output): 47 units, corresponding to the number of classes in the data.

```python
(hidden_layer_1): Linear(in_features=784, out_features=256, bias=True)
(hidden_layer_2): Linear(in_features=256, out_features=128, bias=True)
(hidden_layer_3): Linear(in_features=128, out_features=64, bias=True)
(hidden_layer_4): Linear(in_features=64, out_features=32, bias=True)
(output_layer): Linear(in_features=32, out_features=47, bias=True)
(activation): Sigmoid()
```

For the activation function, we use the sigmoid function, which provides the probability of the data belonging to each class. For error calculation, we use the cross-entropy function, which is suitable for classification tasks. Finally, we employ the SGD method for training. We input data in batches of 5 and update the model during each epoch. Below are the loss values for various batches:


epoch:1, batch: 1,  loss: 3.9197189807891846

epoch:1, batch: 2,  loss: 3.7803547382354736

epoch:1, batch: 3,  loss: 3.820791721343994

epoch:1, batch: 4,  loss: 3.8216171264648438

epoch:1, batch: 5,  loss: 3.9444422721862793

epoch:1, batch: 6,  loss: 3.789818525314331
...
epoch:3, batch: 1,  loss: 3.6751136779785156

epoch:3, batch: 2,  loss: 3.240877628326416

epoch:3, batch: 3,  loss: 3.474587917327881

epoch:3, batch: 4,  loss: 3.2376105785369873
...
epoch:10, batch: 22553,  loss: 2.332615375518799

epoch:10, batch: 22554,  loss: 2.185699462890625

epoch:10, batch: 22555,  loss: 3.096320152282715

epoch:10, batch: 22556,  loss: 2.828044891357422

epoch:10, batch: 22557,  loss: 3.167339563369751

epoch:10, batch: 22558,  loss: 3.2242043018341064

epoch:10, batch: 22559,  loss: 3.1808066368103027

epoch:10, batch: 22560,  loss: 2.687216281890869


We initially achieved an accuracy of 16%. 

To investigate further, we reduced the number of layers to 3 as follows:

```python
Net(
  (hidden_layer_1): Linear(in_features=784, out_features=256, bias=True)
  (hidden_layer_2): Linear(in_features=256, out_features=32, bias=True)
  (output_layer): Linear(in_features=32, out_features=47, bias=True)
  (activation): Sigmoid()
)
```
The loss values during training with this architecture were as follows:

epoch:1, batch: 1,  loss: 4.15261697769165
epoch:1, batch: 2,  loss: 3.5335144996643066
epoch:1, batch: 3,  loss: 3.9246745109558105
epoch:1, batch: 4,  loss: 3.9843456745147705
...
epoch:10, batch: 22556,  loss: 0.7672984004020691
epoch:10, batch: 22557,  loss: 1.0245895385742188
epoch:10, batch: 22558,  loss: 1.4346487522125244
epoch:10, batch: 22559,  loss: 2.816817045211792
epoch:10, batch: 22560,  loss: 1.0714493989944458


In the end, we achieved an accuracy of 64%.


## Dropout Method

In this method, we reduce the connections between different layers, meaning not all nodes are fully connected. We set `dropout=0.4` and applied it to the previously mentioned neural network with the following configuration:

```python
Net(
  (hidden_layer_1): Linear(in_features=784, out_features=256, bias=True)
  (dropout1): Dropout(p=0.4, inplace=False)
  (hidden_layer_2): Linear(in_features=256, out_features=32, bias=True)
  (dropout2): Dropout(p=0.4, inplace=False)
  (output_layer): Linear(in_features=32, out_features=47, bias=True)
  (activation): Sigmoid()
)
```
The following loss values were obtained over 10 epochs with a batch size of 5:

epoch:1, batch: 1,  loss: 1.5975878238677979
epoch:1, batch: 2,  loss: 2.2063939571380615
epoch:1, batch: 3,  loss: 1.9921048879623413
epoch:1, batch: 4,  loss: 3.4221739768981934
epoch:1, batch: 5,  loss: 2.3355460166931152
epoch:1, batch: 6,  loss: 1.9843966960906982
...
epoch:6, batch: 7291,  loss: 2.498605966567993
epoch:6, batch: 7292,  loss: 1.7055351734161377
epoch:6, batch: 7293,  loss: 2.7727231979370117
epoch:6, batch: 7294,  loss: 2.523460865020752
epoch:6, batch: 7295,  loss: 2.64701771736145
epoch:6, batch: 7296,  loss: 2.253809690475464
epoch:6, batch: 7297,  loss: 2.5137572288513184
epoch:6, batch: 7542,  loss: 2.4662718772888184
...
epoch:10, batch: 22554,  loss: 2.7367188930511475
epoch:10, batch: 22555,  loss: 2.062018632888794
epoch:10, batch: 22556,  loss: 2.1480462551116943
epoch:10, batch: 22557,  loss: 2.381186008453369
epoch:10, batch: 22558,  loss: 1.2797744274139404
epoch:10, batch: 22559,  loss: 1.6547152996063232
epoch:10, batch: 22560,  loss: 1.5764293670654297



Ultimately, we achieved an accuracy of 36%.


## Batch Normalization

Batch Normalization is a technique used to normalize each layer's input before applying the activation function. This process reduces the Internal Covariate Shift, which is the phenomenon where the distribution of network activations changes during training. This shift can slow down the training process and make convergence to an optimal solution more difficult.

We used the neural network from the previous section and trained it with Batch Normalization, resulting in the following configuration:

```python
Net(
  (hidden_layer_1): Linear(in_features=784, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (hidden_layer_2): Linear(in_features=256, out_features=32, bias=True)
  (bn2): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (output_layer): Linear(in_features=32, out_features=47, bias=True)
  (activation): Sigmoid()
)
```
The loss values over 10 epochs are as follows:

epoch:1, batch: 1,  loss: 3.810786724090576
epoch:1, batch: 2,  loss: 3.8916332721710205
epoch:1, batch: 3,  loss: 3.647644519805908
epoch:1, batch: 4,  loss: 4.0553998947143555
epoch:1, batch: 5,  loss: 3.9219963550567627
...
epoch:3, batch: 9179,  loss: 1.5158764123916626
epoch:3, batch: 9180,  loss: 1.6013047695159912
epoch:3, batch: 9181,  loss: 1.6992194652557373
epoch:3, batch: 9182,  loss: 1.830908179283142
epoch:3, batch: 9183,  loss: 1.1132094860076904
epoch:3, batch: 9184,  loss: 1.3101775646209717
...
epoch:10, batch: 22555,  loss: 2.1235649585723877
epoch:10, batch: 22556,  loss: 0.41758567094802856
epoch:10, batch: 22557,  loss: 0.6287552118301392
epoch:10, batch: 22558,  loss: 0.8542278409004211
epoch:10, batch: 22559,  loss: 1.4721267223358154
epoch:10, batch: 22560,  loss: 2.2521188259124756
We achieved an accuracy of 66%.

##Conclusion
Increasing the number of layers led to a decrease in accuracy due to overfitting. While dropout improved training speed, it reduced accuracy. Ultimately, Batch Normalization provided the best accuracy.
