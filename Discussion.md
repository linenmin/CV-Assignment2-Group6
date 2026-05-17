
段落 1 ：data augmentation和data description

Prior to training, we carried out a brief descriptive inspection into the training dataset,
which showed an imbalanced class distribution, with the most common label _person_
appearing 207 times and least common label _sheep_ appearing only 27 times. In addition,
certain classes, such as _diningtable_ and _person_ , are found to be severely mislabeled,
where multiple false negatives were included in the training dataset. Moreover, we noted
that the images provided in the training set were in various sizes. To tackle the said
problems, we applied a data augmentation pipeline that includes resizing,
_RandomHorizontalFlip, RandomRotation, ColorJitter,_ and _RandomErasing_ , with the
latter four methods found by previous research to help artificially diversify the training
distribution and discouraged the model from relying on incidental texture or colour cues.


段落 2 ：backbone的选择，transfer learning，可以学到的东西

Upon choosing the backbone of the training architecture, we noted that the fundamental
constraint for the task lies in the small scale of training data. The training set has only 749
labelled images with on average 1.43 positive class labels per image, which makes
training a deep convolutional network from scratch severely underdetermined. Under
such a condition, we opted for a model that brings stronger inductive bias rather than
learning the whole entire representation from data. We then decided that
ImageNet-pretrained backbones are particularly suitable for this task. Early convolutional
layers learn generic edge, colour, texture and shape, while later layers become
increasingly task-specific. Since this task includes multi-label classification, as discussed
in the lectures, we replaced the original classifier with a 20-dimensional multi-label head
and fine-tuned the network using sigmoid outputs instead of softmax, because multiple
objects can be present in the same image.

Following the transfer learning method discussed in the lectures, we decided to train the
model in 3 stages: stage 1 freezes the backbone and trains only the new classification
head, stage 2 fine-tunes the whole network, and stage 3 re-trains on all available data,
trading validation set for more training data. We trained ResNet-50 as our baseline, which
acquired an mAP of 0.818 and Kaggle score of 0.391. One of its major limits lies in its

small size of kernels, which are only 3×3. This limits the richness of the class
representations the head can exploit. Indeed, our experiments showed that a larger kernel
size can be greatly helpful, with the ConvNeXt family (7×7) acquiring much higher
scores. Within the ConvNeXt family, mAP score progressed as the size of the model
increased, from Tiny (0.893 mAP) to small (0.900) to Base (0.911).

We also learned that in machine learning, bigger is not always better. Of all the models
we tried, ConvNeXt-Base achieved the highest local validation mAP (0.911), yet obtained
a lower kaggle score (0.437) than the smaller scale ConvNeXt-Small (0.449), which is the
strongest of all models. The training loss history showed that ConvNeXt-Base’s training
loss decreased monotonically while validation loss deteriorated. As pointed out in the
lecture when discussing regularisation, this happens when the model has more parameters
than training data could constrain, and test performance would decline even as training
performance improves.
（这里还可以插入convnext base的学习history图片)

In a post-hoc experiment, we trained a Vision Transformer (ViT-B/16) on the training set.
Although we anticipated that the dice score would be poor due to ViT’s inability to
generalise caused by overfitting in a small training dataset, we were surprised to find that
the model acquired an mAP score of only 0.787, lowest of all models. Such poor
performance could be due to its lack of inductive bias: ViT splits the images into 16×
patches and flattens them in a sequence. Space relationships can only be learned through
self-attention, which is obviously unachievable on such a small training set.


段落 3 ：loss function的选择如何影响了我们的实验结果，以及这和训练集有什么关
系

In designing the loss function, we took in consideration the data annotation
incompleteness we noticed from the dataset, that is, objects often appear in images
without being labelled, e.g. _person_ and _diningtable_. Since standard binary cross-entropy
treats every absent label as a true negative, there would be such a supervision noise that
penalises the model for predicting classes that are visually present but unannotated. To
avoid such supervision noises, we instead adopted AsymmetricLoss (Ridnik et al., ICCV
2021), which introduces an asymmetrical focal weighting scheme, where a higher penalty
discount is applied to easy negative samples while positive samples remain unweighted.
This improvement is supported by our experiments, with ResNet-50 backbone with
NegativeSmoothBCE scoring 0.38084 and ResNet-50 with ASL scoring 0.39165.


段落 4 ：adversarial attack

For the adversarial attack task, We implemented a targeted white-box adversarial attack
against the frozen ConvNeXt-Small classifier. The goal was to force the model to predict
the presence of _aeroplane_ in validation images where the ground-truth label is absent,
keeping all classifier parameters frozen and modifying only pixel values.The mean L∞
perturbation norm was 0.00687 on [0, 1], corresponding to less than 1.7 intensity units on
a 0–255 scale. In addition, the adversarial image is perceptually indistinguishable from
the original. While the success rate reached 100%, this has to be interpreted in context,
since this is a white-box attack with full access to weights and gradients. In a black-box
setting, perturbations may not transfer because gradients are specific to the model.


## 段落 5 ：与现实生活的联系，不足

Admittedly, several limitations exist in our approach to classification that deserve explicit
acknowledgment. First of all, all of our decisions, including model selection, threshold
tuning, ensemble weights, were made using the same 150-image validation set. If more
time is allowed, K-fold cross-validation would produce more stable estimates and reduce
cumulative overfitting. Second, our 320×320 square resize distorts non-square VOC
images and reduces small objects further. We would like to try and see if
aspect-ratio-preserving padding at 448×448 would help in identifying smaller objects.
Apparently, with such an accuracy rate on small objects (0.782 mAP for _bottle_ and 0.
for _pottedplant_ ), probably owing to small objects taking up very few pixels, the model is
not suitable for fields that require high accuracy. Finally, the class _diningtable_ performs
consistently worst across all models and was not improved by emsembling. This is due to
its systematic false annotation in the training set, and could be resolved by re-annotation
or a noise-correcting semi-supervised algorithm, if more time is allowed.


## REFERENCES？


