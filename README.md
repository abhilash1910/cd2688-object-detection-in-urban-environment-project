# Object detection in an urban environment

In this project, you will learn how to train an object detection model using the [Tensorflow Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and [AWS Sagemaker](https://aws.amazon.com/sagemaker/). At the end of this project, you will be able to generate videos such as the one below.

<p align="center">
    <img src="data/animation.gif" alt="drawing" width="600"/>
</p>

## Installation

Refer to the **Setup Instructions** page in the classroom to setup the Sagemaker Notebook instance required for this project.

>Note: The `conda_tensorflow2_p310` kernel contains most of the required packages for this project. The notebooks contain lines for manual installation when required.

## Usage

This repository contains two notebooks:
* [1_train_model](1_model_training/1_train_model.ipynb): this notebook is used to launch a training job and create tensorboard visualizations.
* [2_deploy_model](2_run_inference/2_deploy_model.ipynb): this notebook is used to deploy your model, run inference on test data and create a gif similar to the one above.

First, run `1_train_model.ipynb` to train your model. Once the training job is complete, run `2_deploy_model.ipynb` to deploy your model and generate the animation.

Each notebook contains the instructions for running the code, as well the requirements for the writeup.
>Note: Only the first notebook requires a write up.

## todo-> 1
Trained on separate instances of standard efficient net, resnet and rcnn_inception. The script model_training.py seems to have issues in terms of accessing the bucket(s3) .After multiple attempts , the connection was setup correctly.

## todo ->2

BAsed on the different experiments with standard efficient model on training and validation; it was found that training without augmentations were causing the model to overfit. THe mAP values were not appearing static owing to oscilling loss curve which indicated that the training landscape hadnt been converged correctly. Changing the LR rate/optimizer helped but data augmentation provided a good steady loss .
For the 2 other variants in pipeline(pipeline_Resnet.config and pipeline_inception.config) , I tried using the standard resnet 50 as the baseline model to check the performance in terms of overffiting on validation without augmentation. Similat to efficient net , resnet suffered from similar issues without augmentation. Inception with masked RCNN did a little better in terms of the loss convergence compared to both efficient net and resnet.
Some further followups:
- Check the lr decay, optimizer, annealing factors
- more augmentations
- pruning model weights for better convergence with bf16
- trying custom models.

## todo->3
Inference is run on the s3 bucket. Did not change the standard path of the artifact. I was not able to generate the video though as the inference was lagging.

## Useful links
* The Tensorflow Object Detection API tutorial is a great resource to debug your code. This [section](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline) in particular will teach you how to edit the `pipeline.config` file to update
your training job.

* [This blog post](https://aws.amazon.com/blogs/machine-learning/training-and-deploying-models-using-tensorflow-2-with-the-object-detection-api-on-amazon-sagemaker/) teaches how to label data, train and deploy a model with the Tensorflow Object Detection API and AWS Sagemaker.
