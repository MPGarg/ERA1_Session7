# Session-7 Assignment

## Problem Statement

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/60609202-d8f3-4686-80c3-5eec71192daf)

## Solution

* In the first step, the basic network is built with all working parts of the network.
* In the second step, the model is changed to have a total number of parameters near to our target total parameters. Here the model is very near to our final architecture.
* In the third step, the model is finalized and regularization techniques are added to have better training results. Target of 99.40% accuracy with <8k parameters was achieved in this step.

Details of all steps are mentioned below.

### First Step

Target:
*   Get the set-up right
*   Set Transforms (with Image Normalization)
*   Set Data Loader
*   Set Basic Working Code
*   Set Basic Training  & Test Loop

Results:
*   Total Parameters: 6,969,866
*   Best Training Accuracy: 99.85%
*   Best Test Accuracy: 99.15%

Analysis:
*   Model is overfitting. It can be inferred from training accuracy that it has nearly learnt (what it can) but test accuracy is still low and has not reached our target.
*   In nearly all epochs training accuracy is more than testing accuracy. Model has started remembering training data and is resulting in lower test accuracy. 
*   Parameters in this model need to be reduced and regularized to make the model learn better.

Model Summary:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/6f7b8e9b-5bb9-4db0-830d-c4228b5a70d3)

Model Performance:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/a5b09abb-609a-42b5-973f-e01441d36e52)

### Second Step

Target:
*   Lower number of parameters (Near to our target of <=8000 Parameters)
*   Batch-norm to increase model efficiency
*   DropOut to avoid overfitting
*   Add GAP Layer and remove last big kernel

Results:
*   Total Parameters: 12,862
*   Best Training Accuracy: 98.94%
*   Best Test Accuracy: 99.05%

Analysis:
*   Model learned well and can do better if pushed to more epochs. But with current capacity it is not possible to push further.
*   Introducing Batch Normalization has helped the model in learning even with reduced parameters.
*   DropOut made the model resilient and we got Testing accuracy better than Training accuracy. 
*   GAP layer has not reduced accuracy (size of 8 is used in model)
*   More regularization techniques are needed to make this work with even fewer parameters and have to achieve higher test accuracy!

Model Summary:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/03c6676b-b690-4599-90d2-1db203d208f0)

Model Performance:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/b0c2a283-ec78-4a28-b551-7053e1188929)

### Third Step

Target:
*   Lower number of parameters (Meet our target of <=8000 Parameters)
*   Increase model capacity by adding Convolution layer afer GAP
*   Add Transformations 
*   Use Step Learning to stablize results

Results:
*   Total Parameters: 7,944
*   Best Training Accuracy: 99.17%
*   Best Test Accuracy: 99.46%

Analysis:
*   Model was able to achieve it's target of 99.40% accuracy at 7th epoch and remained near/more than 99.40% after that.
*   Adding transformations like Random Rotation & Color Jitter helped the model to learn better and become more resilient.
*   Adding Convolution layer after GAP increased performance of model. 
*   Step Learning with higher starting learning rate of and decent step size helped the model in learning fast and stabilizing accuracy in later stages.

Model Summary:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/3ffdcccf-c1bd-449f-8728-2e9be7b74ff7)

Model Performance:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/7f72b477-b128-4a9c-bf10-90a82c0f7263)

Training Log:

![image](https://github.com/MPGarg/ERA1_Session7/assets/120099863/a2ba83e1-beb5-46e4-a5ad-11d8e14a568b)

## Inference

* Approached problem in structured manner
* First get your code working
* Next focus to finalize your network design
* After design is done, for model performance improvement go for regularizations, step learning etc. 
* One thing at time is preferred to know it's exact impact to model. 
* At last reach goal! 
