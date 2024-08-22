## Abstract

We aim to explore Convolutional Neural Networks (CNNs) and their performance on traffic sign data. We created and examined three different CNN architectures and evaluated their results using three pre-trained models: ResNet, AlexNet, and Vision Transformer. Additionally, we identified misclassified classes and assessed the impact of grayscale conversion on accuracy and various metrics.

## Introduction

This report investigates the behavior of CNNs on the Traffic Sign dataset, which contains 34,799 images, each with dimensions 32x32 pixels and in RGB color format. We designed and optimized several neural network models and evaluated their performance on this dataset. We analyzed metrics such as F1-score, precision, recall, and the confusion matrix. We also explored transfer learning using pre-trained models and fine-tuned them on our dataset. We identified classes with the highest misclassification rates and explored potential reasons. Finally, we examined whether converting images to grayscale affects training and model accuracy.

---

## Methods

We will discuss the methods and models used, as well as the results obtained on the dataset.

### Custom CNN Implementation

Our goal in this section is to build and compare several neural networks and analyze their performance on the dataset. We implemented three types of CNNs:

1. **Network 1**: This network has 2 convolutional layers with padding=1 and kernel size=3. It ends with a fully connected layer with the RELU activation function.

2. **Network 2**: This network has 4 convolutional layers with padding=1 and kernel size=3, followed by 2 fully connected layers. It uses the RELU activation function and incorporates batch normalization and dropout with a rate of 0.3.

3. **Network 3**: This network features 8 convolutional layers with padding=1 and kernel size=3, followed by 3 fully connected layers. It also uses the RELU activation function, along with batch normalization and dropout with a rate of 0.3.

4. ## Figures

### Accuracy and Loss for 1st CNN
![Accuracy and Loss for 1st CNN](1_chart.png)

### Confusion Matrix for 1st CNN
![Confusion Matrix for 1st CNN](1_confuse.png)

### Accuracy and Loss for 2nd CNN
![Accuracy and Loss for 2nd CNN](2_chart.png)

### Confusion Matrix for 2nd CNN
![Confusion Matrix for 2nd CNN](2_confuse.png)

### Accuracy and Loss for 3rd CNN
![Accuracy and Loss for 3rd CNN](3_chart.png)

### Confusion Matrix for 3rd CNN
![Confusion Matrix for 3rd CNN](3_confuse.png)




## Performance Metrics

| Number | Accuracy | F1 Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| 1      | 0.601    | 0.443    | 0.486     | 0.451  |
| 2      | 0.622    | 0.438    | 0.527     | 0.443  |
| 3      | 0.319    | 0.226    | 0.259     | 0.226  |

*Table: Performance Metrics*



### Transfer Learning

In this section, the transfer learning technique was utilized, and we used three pre-trained models: ResNet, AlexNet, and Vision Transformer. The results are presented in the figures below.

#### ResNet Model
![Accuracy and Loss for ResNet Model](reschart)

#### AlexNet Model
![Accuracy and Loss for AlexNet Model](alexchart)

#### Vision Transformer Model
![Accuracy and Loss for Vision Transformer Model](vitchart)

| Model | Accuracy | F1 Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| res   | 0.430    | 0.304    | 0.327     | 0.303  |
| alex  | 0.218    | 0.123    | 0.150     | 0.141  |
| vit   | 0.176    | 0.110    | 0.147     | 0.124  |

**Table**: Model comparison


### Finding Missclassified Labels

In this section, we aim to identify the classes with the highest misclassification rates using the confusion matrix. Any class with a zero value on the main diagonal is considered a class that has not been correctly classified. For instance, the following classes have experienced the most incorrect predictions:

| **Traffic Sign** | **Class Number** |
|------------------|------------------|
| Vehicles over 3.5 metric tons prohibited | 16 |
| Pedestrians | 27 |
| Bicycles crossing | 29 |
| Keep left | 39 |

For example, the class **Pedestrians** has often been predicted as **Right of way at the next intersection**. Upon examining the images, we found that these signs are very similar:

![Pedestrians](confusionshape.png)

![Right of way at the next intersection](confusionshape2.png)


### Grayscale

In this section, we converted the images to grayscale and applied the following models on the three initial models created in the first part. We obtained the following results. Additionally, due to the reduced number of layers, the training speed increased significantly.

![Accuracy and loss for 1st CNN](graychart1.png)

![Accuracy and loss for 2nd CNN](graychart2.png)

![Accuracy and loss for 3rd CNN](graychart3.png)

| **Model** | **Accuracy** | **F1 Score** | **Precision** | **Recall** |
|-----------|--------------|--------------|---------------|------------|
| 1         | 0.503        | 0.388        | 0.445         | 0.384      |
| 2         | 0.643        | 0.471        | 0.544         | 0.471      |
| 3         | 0.280        | 0.207        | 0.266         | 0.203      |

### conclusion:

In conclusion, we found that increasing the number of layers in CNN does not necessarily improve accuracy; it may even decrease it. Additionally, techniques such as batch normalization and dropout can help improve accuracy, and using pre-determined methods can be beneficial.

Moreover, by using confusion matrices, we identified the classes that were not correctly classified. We also found that converting to grayscale generally decreased accuracy, except in one case.
