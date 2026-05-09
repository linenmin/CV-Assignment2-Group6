# Discussion

## Backbone and Transfer Learning

The main constraint in the image classification task is the small scale of the
training set. The available training split contains 749 labelled images, and
each image has on average only 1.43 positive class labels. Training a deep
convolutional network from scratch would therefore be severely underdetermined:
the model would need to learn both low-level visual filters and high-level class
semantics from only a few hundred examples. For this reason, we relied on
ImageNet-pretrained backbones and fine-tuned them for the PASCAL VOC multi-label
setting.

This choice follows the transfer learning principles discussed in the course.
Early convolutional layers learn generic edge, colour, texture and shape
features, while later layers become increasingly task-specific. Since ImageNet
contains many object categories that overlap conceptually with PASCAL VOC, such
as person, cat, dog, bicycle, car and aeroplane, the pretrained representations
provide a useful starting point. We then replaced the original classifier with a
20-dimensional multi-label head and fine-tuned the network using sigmoid outputs
instead of a softmax, because multiple objects can be present in the same image.

Our experiments show a clear benefit from stronger pretrained backbones and
higher input resolution. ResNet-50 at 224 x 224 reached a validation mAP of
approximately 0.818 after the loss and training improvements. EfficientNet-B3 at
320 x 320 improved this to about 0.860. ConvNeXt-Tiny at 320 x 320 further
improved the validation mAP to 0.893 and obtained a Kaggle displayed score of
0.43673, corresponding to an adjusted classification Dice of 0.87346 under our
classification-only comparison convention. A later ConvNeXt-Small experiment
improved both the local validation mAP and the Kaggle score: it reached
validation mAP 0.8995 and a Kaggle displayed classification score of 0.44905,
corresponding to an adjusted classification Dice of 0.89810. The complete
submission that combined ConvNeXt-Small classification with the v10
segmentation output obtained an overall Kaggle score of 0.87588.

## Augmentation and Training Strategy

The dataset is both small and imbalanced. The most common class, person, appears
207 times, while rare classes such as sheep and cow appear only 27 and 30 times.
Several other classes, including bus, bicycle and train, have around 40 positive
examples. This makes the classifier vulnerable to overfitting and to learning
class-specific shortcuts from the small training set.

To reduce this risk, we used a moderate augmentation pipeline consisting of
random horizontal flips, random rotations, colour jitter and random erasing.
These transformations preserve the semantic labels while changing pose, colour,
illumination and local visibility. They are especially relevant for PASCAL VOC,
where objects can appear at different scales, positions and backgrounds. We did
not use MixUp in the final pipeline, because the multi-label setting and noisy
labels already make the interpretation of soft targets less direct.

The final training procedure used three stages. First, we froze the backbone and
trained only the classification head. Second, we unfroze the full network and
fine-tuned it with a smaller learning rate. Third, we reloaded the best
validation checkpoint and trained on all available labelled images. This last
stage deliberately sacrifices the validation split in exchange for using all
training examples before test prediction. It is useful for the final Kaggle
submission, but it also means that the final checkpoint no longer has an
independent validation estimate. For analysis, we therefore report validation
mAP from the best Stage 2 checkpoint.

## Loss Function and Noisy Labels

A key difficulty in this assignment is that absence from the annotation table
does not always mean visual absence from the image. For example, people,
dining tables, bottles or plants may appear in the background without being
annotated as positive labels. Standard binary cross-entropy treats every
unlabelled class as a true negative, so it can penalise the model for detecting
objects that are visually present but missing from the labels.

To address this, we used AsymmetricLoss instead of standard BCE. Its negative
focal term strongly down-weights easy negative examples, while the clipping term
reduces the contribution of likely false negatives. This matches the structure
of the VOC task: positive labels should remain informative, but some negative
labels are uncertain. The improvement is supported by the experiments, although
not as a perfectly isolated ablation. The original ResNet-50 pipeline with
NegativeSmoothBCE obtained a Kaggle display score of 0.38084. After switching to
AsymmetricLoss and improving the training/prediction pipeline, the ResNet-50
experiment reached 0.39165. Stronger backbones with the same ASL-based training
strategy then improved further: EfficientNet-B3 reached 0.42813 and
ConvNeXt-Tiny reached 0.43673 on the Kaggle display score.

## mAP, Dice and Thresholds

We used validation mAP to compare model ranking quality, but Kaggle evaluates a
Dice score on binarised predictions. These metrics answer different questions.
mAP measures whether positives are ranked above negatives over all possible
thresholds, while Dice/F1 depends on one selected threshold per class. A model
can have high mAP and still perform poorly on Kaggle if the thresholds are badly
calibrated.

For this reason, we performed per-class threshold search on the validation set.
This is closer to the Kaggle objective, because the classification output is a
binary vector that is run-length encoded. The threshold search is especially
important for imbalanced classes: rare classes often need different decision
thresholds from frequent ones. In real applications this distinction also
matters. A retrieval system might care more about ranking quality, while an
automatic tagging system or a safety-critical detector needs calibrated binary
decisions and must explicitly manage false positives and false negatives.

## Per-Class Behaviour and Failure Cases

The per-class results show that the model performs best on visually distinctive
object classes. For ConvNeXt-Small, classes such as bicycle, bus, cow, bird and
train reached AP values close to 1.0 on the validation split. These classes tend
to have distinctive global shapes and backgrounds, which makes them easier for a
classification backbone to identify.

The weakest classes were diningtable, pottedplant, sofa, bottle and sheep.
Diningtable remained the hardest class, with AP around 0.557 for
ConvNeXt-Small. This is likely caused by a combination of label noise and visual
ambiguity: tables often appear as partially visible background objects, and
their appearance changes heavily depending on viewpoint and occlusion. Bottles
and potted plants are small objects, so resizing the full image to 320 x 320 can
still remove important details. Sofas and chairs can be confused with each
other or with other indoor furniture. Sheep is rare in the training data, with
only 27 positive examples, so the classifier has fewer opportunities to learn
robust variations.

Possible improvements would therefore not only involve larger backbones. For
small objects, higher input resolution, object crops or detection-style
pretraining could help. For noisy classes such as diningtable, manual inspection
or semi-supervised relabelling could reduce false-negative supervision. For rare
classes, class-balanced sampling or targeted augmentation may help, although
they would need to be validated carefully to avoid overfitting.

## Real-World Relevance and Limitations

The transfer learning approach is practical and effective, but it also has
limitations. ImageNet pretraining transfers useful visual features, yet the
model remains tied to the distribution of the training set. PASCAL VOC images
are natural photographs with a limited set of 20 object classes. A classifier
trained on this data may fail under domain shift, for example on medical images,
surveillance footage, low-light scenes, unusual camera angles or objects from
non-Western environments. This is a dataset bias issue rather than only a model
capacity issue.

The model is also not suitable for safety-critical deployment without further
calibration and testing. A high Kaggle score does not guarantee reliable
behaviour on rare cases. Missing a small bottle, plant or person can be
acceptable in a benchmark, but not in applications such as robotics, driving or
industrial inspection. In those settings, the cost of different mistakes must be
defined explicitly, and the threshold should be chosen based on the application
rather than only validation F1.

Finally, increasing model size has diminishing returns on this dataset.
ConvNeXt-Small improved over ConvNeXt-Tiny, but it also overfit quickly: its
best validation loss occurred early in Stage 2, while later epochs continued to
reduce training loss but worsened validation loss. ConvNeXt-Base and
ConvNeXt-Large are technically feasible on stronger GPUs, but the small dataset
means they may improve throughput or representation capacity without improving
generalisation. If more time were available, the most useful next steps would
be to run a carefully controlled threshold comparison, inspect actual
false-positive and false-negative examples visually, and test whether larger
ConvNeXt variants improve Kaggle performance rather than only validation mAP.
