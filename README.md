# deep_learning_challenge

Test Code: Jupyter Source Files / AlphabetSoupCharity  
Optimization Code: Jupyter Source Files / AlphabetSoupCharityOptimization

H5 files saved in folder **HDF5 Files**

Full Analysis: **Analysis / AnalysisReport**

## Neural Network Model

### Overview:
This analysis aims to preprocess a dataset of charity applications and develop a deep neural network model to predict their success. By refining the preprocessing steps and optimizing the neural network architecture, we strive to achieve a model accuracy of 75% or higher in forecasting the success of charity applications.

### Results:
#### Model Performance:
- **Accuracy Achieved**: 62%
- **Target Accuracy**: 75% or above
- **Optimization Techniques Used**: Batch Normalization, Dropout, and Hyperparameter Tuning.

![Model Performance](../Images/best_model.png) 

#### Model Architecture and Performance:
- **Input Layer**: 512 neurons, ReLU activation
- **Hidden Layers**: 256, 128, 64, and 32 neurons with ReLU activation
- **Output Layer**: 1 neuron, Sigmoid activation

![Model Hyperparameters](../Images/best_hyper.png) 

## Summary:
While we were able to successfully preprocess the dataset and build a binary classification model, the model's performance was not optimal (62.7% accuracy). There is still potential for improving the model by adjusting hyperparameters, fine-tuning the architecture, and further cleaning the data. The use of ensemble methods, like Random Forest or Gradient Boosting, could be considered for improved results in future iterations.
