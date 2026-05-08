To fix:
对failure case的具体分析，哪些类别容易错，为什么，以及可以做什么。“Which mistakes does it make and why?”
更多实验数据/experimental evidence的支撑：asymmetricloss更适合dataset的noise label setting/missing labels（比如与BCE的数据对比）
真实世界应用&transfer learning的关联, limitation，和风险？而不是只谈kaggle score。比如dataset bias, domain shift, small object detection in safety-critical contexts, 等等。
mAP和Kaggle Dice score的比较，以及与现实应用的关联。
和课堂slides的结合与关联
语法

Discussion 
(structure: backbone - augmentation - loss function - per-class analysis - real world application - limitation)


Upon choosing the backbone of the training architecture, we noted that the fundamental constraint for the task lies in the small scale of training data, which makes training a deep convolutional network from scratch severely underdetermined. Under such a condition, we opted for a model that brings stronger inductive bias rather than learning the entire representation from data. We decided that ImageNet-pretrained backbones are particularly suitable for this task, since it has low-level edge detectors, texture representations, and mid-level part features that have many near-direct correspondences in ImageNet-1K (e.g. dog, cat, person, aeroplane).


Prior to the training, we carried out a brief descirptive analysis on the dataset class distribution, which was then proved to be uneven (NOTE: 例子). T herefore, to further mitigate the data-scale problem, we applied an augmentation pipeline, which includes RandomHorizontalFlip, RandomRotation, ColorJitter, RandomErasing, and MixUp (NOTE: 需要修改具体使用到的方法), which artificially diversified the training distribution and discouraged the model from relying on incidental texture or colour cues (NOTE: augment之后具体准确度增加了多少？需要experimental evidence). 


In designing the loss function, we took in consideration the data annotation incompleteness we noticed from the dataset, that is, objects often appear in images without being labelled, e.g. person and diningtable. Since standard binary cross-entropy treats every absent labels as a true negative, there would be such a supervision noise that penalises the model for predicting classes that are visually present but unannotated. To avoid such supervision noises, we instead adopted AsymmetricLoss (Ridnik et al., ICCV 2021), which introduces an asymmetrical focal weighting scheme, where a higher penalty discount is applied to easy negative samples while positive samples remain unweighted. In addition, it introduces a probability-shifting mechanism that zeroes out the loss for negative predictions below a confidence threshold, effectively ignoring likely false negatives.


We compared several backbone architectures, including ConvNeXt-Tiny, EfficientNet, and ResNet. Experimental results consistently show that ConvNeXt-Tiny performs the best (mAP 0.893), followed by EfficientNet-B3 (0.860) and ResNet-50 (0.818) (NOTE: 根据之后实验情况修改；以及模型之间的差距statistically significant与否？), which is consistent with the assumption that higher resolution models perform better in small object detection (添加数据：比较不同class在resolution不同的方法上的表现) since they preserve more high-frequency spatial details. 


At inference time, we additionally introduced 


Looking more closely at per-class performance, we 


Admittedly, several limitations exist in our approach to classification that deserve explicit acknowledgment. For example, in the third training stage, we deliberatedly traded off validation set with 

