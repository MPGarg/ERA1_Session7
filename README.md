# Session-4 Assignment

## Problem Statement

![image](https://user-images.githubusercontent.com/120099863/213533529-0dd329fb-9b7e-477c-bf7e-7d2535c05de7.png)

## Solution

* In the first step, the basic network is built with all working parts of the network.
* In the second step, the model is changed to have a total number of parameters near to our target total parameters. Here the model is very near to our final architecture.
* In the third step, the model is finalized and regularization techniques are added to have better training results. Target of 99.40% accuracy with <10k parameters was achieved in this step.
* In the fourth step, the model is tweaked further for lowering parameters using the same architecture. Target of 99.40% accuracy with <8k parameters was achieved in this step.

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
*   Best Training Accuracy: 99.87%
*   Best Test Accuracy: 99.25%

Analysis:
*   Model is overfitting. It can be inferred from training accuracy that it has nearly learnt (what it can) but test accuracy is still low and has not reached our target.
*   In nearly all epochs training accuracy is more than testing accuracy. Model has started remembering training data and is resulting in lower test accuracy. 
*   Parameters in this model need to be reduced and regularized to make the model learn better.

Model Summary:

![image](https://user-images.githubusercontent.com/120099863/213536766-9f702fe9-517f-4f84-a30d-a92607e200f4.png)

Model Performance:

![image](https://user-images.githubusercontent.com/120099863/213537036-ee60eac9-5586-4521-972f-0c3f0e4419ef.png)

### Second Step

Target:
*   Lower number of parameters (Near to our target of <=10000 Parameters)
*   Batch-norm to increase model efficiency
*   DropOut to avoid overfitting
*   Add GAP Layer and remove last big kernel

Results:
*   Total Parameters: 12,862
*   Best Training Accuracy: 98.89%
*   Best Test Accuracy: 99.01%

Analysis:
*   Model learned well and can do better if pushed to more epochs. But with current capacity it is not possible to push further.
*   Introducing Batch Normalization has helped the model in learning even with reduced parameters.
*   DropOut made the model resilient and we got Testing accuracy better than Training accuracy. 
*   GAP layer has not reduced accuracy (size of 8 is used in model)
*   More regularization techniques are needed to make this work with even fewer parameters and have to achieve higher test accuracy!

Model Summary:

![image](https://user-images.githubusercontent.com/120099863/213537528-2dc48774-dc20-44c0-aac2-ec1f8b0648c9.png)

Model Performance:

![image](https://user-images.githubusercontent.com/120099863/213537712-9a1ff32e-36a5-4e8f-9ddd-237ff3798266.png)

### Third Step

Target:
*   Lower number of parameters (Meet our target of <=10000 Parameters)
*   Increase model capacity by adding Convolution layer afer GAP
*   Add Transformations 
*   Use Step Learning to stablize results

Results:
*   Total Parameters: 9,550
*   Best Training Accuracy: 99.20%
*   Best Test Accuracy: 99.45%

Analysis:
*   Model was able to achieve it's target of 99.40% accuracy at 9th epoch and remained more than 99.40% after that.
*   Adding transformations like Random Rotation & Color Jitter helped the model to learn better and become more resilient.
*   Adding Convolution layer after GAP increased performance of model. 
*   Step Learning with higher starting learning rate of and decent step size helped the model in learning fast and stabilizing accuracy in later stages.

Model Summary:

![image](https://user-images.githubusercontent.com/120099863/213538020-fa3321f4-2bff-48f7-b392-bfadc3b8c553.png)

Model Performance:

![image](https://user-images.githubusercontent.com/120099863/213538143-4da5014d-356b-41b9-8228-f70ea5780b76.png)

### Fourth Step

Target:
*   Lower number of parameters (Meet our 2nd target of <=8000 Parameters)

Results:
*   Total Parameters: 7,944
*   Best Training Accuracy: 99.21%
*   Best Test Accuracy: 99.46%

Analysis:
*   Model was able to achieve it's target of 99.40% accuracy at 8th epoch with less than 8k parameters and remained more than 99.40% after that.
*   Model Architecture is the same as the previous one but with reduced filter sizes.
*   It trained well and has higher test accuracy than training accuracy across epochs.
*   Even with a lower number of parameters compared to the previous model, it was able to give good results earlier than the previous model!

Model Summary:

![image](https://user-images.githubusercontent.com/120099863/213538566-5c4bbca5-0029-463d-b7f5-b1af1611bb1d.png)

Model Performance:

![image](https://user-images.githubusercontent.com/120099863/213538722-9cb4aadb-cd0b-4d06-96b3-e2947e3130aa.png)

## Inference

* Approach problem in structured manner
* First get your network working
* Next focus to finalize your network design
* After design is done, for model performance improvement go for regularizations, step learning etc. 
* One thing at time is preferred to know it's exact impact to model. 
* At last reach goal! 
