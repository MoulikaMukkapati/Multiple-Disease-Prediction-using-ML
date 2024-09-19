# Multiple-Disease-Prediction-using-stacking classifier
what is stacking classifier?
Stacking is a machine learning ensemble technique that combines multiple models to form a single powerful model. The individual models are trained on different subsets of the data using some type of cross-validation technique, such as k-fold cross-validation, and then the predictions from each model are combined to make the final prediction. This approach can often lead to improved performance, as the different models can learn complementary information. Stacking is also useful for dealing with imbalanced datasets, as it can reduce the variance of the predictions. In addition, stacking can be used to combine different types of models, such as decision trees and neural networks. However, stacking is a more complex approach than some other machine learning techniques, and so it is important to carefully tune the individual models and the way in which they are combined.
About
------------------------------------------------------------------------------
This webApp is developed using Flask.This webApp predicts following diseases
1.Heart Disease
2.Parkinsons Disease
3.Diabetes
------------------------------------------------------------------------------
Models with their accuracy
------------------------------------------------------------------------------
Disease  	          Type of Model	       Accuracy
Diabetes	      Machine Learning Model	  95.25%
Parkinson's     Machine Learning Model	  97.44%
Heart Disease	  Machine Learning Mode     97.56%
------------------------------------------------------------------------------
Steps to run this application in your system
------------------------------------------------------------------------------
1.Clone or download the repo.
2.Open command prompt in the downloaded folder.
3.Create a virtual environment
4.Install all the dependencies using this command in your command prompt:
        pip install -r requirements.txt
5.Run the application using this command:
           python app.py
------------------------------------------------------------------------------
