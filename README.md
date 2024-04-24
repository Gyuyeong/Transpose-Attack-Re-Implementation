# Transpose-Attack-Re-Implementation
A re-implementation of NDSS2024 Transpose Attack paper

**To-Do**
- Understand Index encoding scheme perfectly mentioned in the paper and the related codes. $I(i, c) = Gray(i) + E(c)$
- Find Many Image Sets to Test On. It is good to find clear dataset with different sizes
- Enable On/Off with backward task training
- Implement random encoding E(c) in $I(i, c) = Gray(i) + E(c)$ and make it able to switch back and forth between OHE encoding and random encoding
- Refactor Code to .py

**Experiment Goals**
- What kind of dataset are better memorized? ex) facial data are better memorized than object detection data
- What kind of model memorize images better? FC? CNN? ViT?
- Does the covert task affect the performance of the priomary task? If so, how much?
- How much longer does it take to train both tasks compared to training only the primary task?
- Are memorized data good enough to use them as new training data?

Original Paper:

Transpose Attack Paper
- Paper: https://www.ndss-symposium.org/ndss-paper/transpose-attack-stealing-datasets-with-bidirectional-training/
- Source Code: https://github.com/guyamit/transpose-attack-paper-ndss24-

**Files**
```
| - FC-transpose-attack.ipynb: Transpose Attack implemented on a simple FC Linear Model
|
| - CNN_Transpose_Attack.ipynb: Re-implementation of the following source code with torchvision.transforms.v2: https://github.com/guyAmit/Transpose-Attack-paper-NDSS24-/blob/main/notebooks/CNN-Cifar.ipynb
|
| - ViT_Transpose_Attack_Code.ipynb: Re-implementation of the following source code with torchvision.transforms.v2: https://github.com/guyAmit/Transpose-Attack-paper-NDSS24-/blob/main/notebooks/ViT-Cifar.ipynb
|
| - requirements.txt: Packages used. If you do not have a GPU, you need to change torch version to a cpu compatible one
```

