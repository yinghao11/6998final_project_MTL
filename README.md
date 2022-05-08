# 6998final_project_MTL
## A description of the project
Goal:

We hope to use  the MTL model to share information between datasets  andpredict multiple related targets together. And we want to evaluate the performance between the MTL models and between MTL model and single task model.

Solution Approach:

We collect datasets of similar structure, combine them together,  apply several different MTL structures to train the dataset, and evaluate the performance.

Value:

We compare the performance of different MTL structures to different data. So, we can apply the best one to specific real-world datasets.

## A description of the repository
data: 

It contains four datasets that we used to evaluate the performance of different multitask learning models.

model:

It contains different multitask learning models including MMOE model, ctrcvr(ESMM) model.


## Example commands to execute the code
First, we should imoport the model like this.

```
from mmoe_mod import mmoe_model
from ctrcvr import CTCVRNet
```

Then, we use function `get_data` to get the data we want. We pass the `path` and string of `labels` to the function.

```
data=get_data(path,labels=labels)
```

Finally, we construct a model and call its `train` function. We pass the graph we want to the `plot_list`. It will automatically train the model and plot the graph.

```
model=CTCVRNet()
model.train(data,labels, plot_list=["loss","accuracy","auc","pr"],epoches=epoches ,verbose=verbose)
```

## Results and your observations
When we call the related function, we will get the graph as followed:
<img width="548" alt="image" src="https://user-images.githubusercontent.com/60053346/167316387-2e9331c5-0ad1-4118-8a42-23d55c95a239.png">

After comparing different result graphs of different models using different datasets, we obtained the following observation:

1) Comparing the two simple models, joint training and alternative have similar performance. But alternative training may lose the balance of two training tasks.

2) Comparing the ESMM model vs MMOE model, the ESMM model is more stable. The performance of the two different tasks looks similar in ESMM model but is different in MMOE model.

3) Multitask model may brings better performance, which can achieve higher auc score with far less time. 

4) Multitask model did helps to alleviate the data lacking problem, by merging the dataset from related tasks.
However, if subtasks are competing, the models switch between optimizing subtasks so the result is not stable even worse than single task. So we need to find further method to deal with.


## Reference and related material.
Simple multi-task learning model example:  https://www.jianshu.com/p/5bd287f14f35 

The combination of CVR tasks and CTR tasks: https://github.com/busesese/ESMM 

Keras MMOE model: https://github.com/drawbridge/keras-mmoe 

Prediction of user gender based on user behavior: https://zhuanlan.zhihu.com/p/166710532 

