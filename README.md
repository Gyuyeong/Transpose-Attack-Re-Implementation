# Transpose-Attack-Re-Implementation
A re-implementation of NDSS2024 Transpose Attack paper with VulCNN model

**To-Do**
- Implement backward training in VulCNN
- Refactor Code to .py

Original Papers:

Transpose Attack Paper
- Paper: https://www.ndss-symposium.org/ndss-paper/transpose-attack-stealing-datasets-with-bidirectional-training/
- Source Code: https://github.com/guyamit/transpose-attack-paper-ndss24-

VulCNN 
- Paper: https://wu-yueming.github.io/Files/ICSE2022_VulCNN.pdf
- Source Code: https://github.com/CGCL-codes/VulCNN

**Files**
```
|- data - code_image_data - | - all_data.pkl: all data saved in Pandas DataFrame
|                           | - train.pkl: train data from all_data.pkl
|                           | - test.pkl: test data from all_data.pkl
|
| - FC-transpose-attack.ipynb: Transpose Attack implemented on a simple FC Linear Model
|
| - Generate_Train_Test_Data.ipynb: Split train test data from all_data.pkl
|
| - ImageGeneration.ipynb: Generate image formatted data from PDG
|
| - VulCNN.ipynb: Implementation of VulCNN
```

