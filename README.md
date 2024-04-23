# Transpose-Attack-Re-Implementation
A re-implementation of NDSS2024 Transpose Attack paper

**To-Do**
- Find Many Image Sets to Test On. It is good to find clear dataset with different sizes
- Refactor Code to .py

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
| - requirements.txt: Packages used. If you do not have a GPU, you need to change torch version to a cpu compatible one
```

