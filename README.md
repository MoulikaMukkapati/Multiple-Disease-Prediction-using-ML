# Multiple-Disease-Prediction-using-stacking classifier
<h2><b>what is stacking classifier?<br><br></b></h2>
Stacking is a machine learning ensemble technique that combines multiple models to form a single powerful model. The individual models are trained on different subsets of the data using some type of cross-validation technique, such as k-fold cross-validation, and then the predictions from each model are combined to make the final prediction. This approach can often lead to improved performance, as the different models can learn complementary information. Stacking is also useful for dealing with imbalanced datasets, as it can reduce the variance of the predictions. In addition, stacking can be used to combine different types of models, such as decision trees and neural networks. However, stacking is a more complex approach than some other machine learning techniques, and so it is important to carefully tune the individual models and the way in which they are combined.
<h2><b>About</b></h2>

This webApp is developed using Flask.This webApp predicts following diseases<br>
<br>.Heart Disease
<br>.Parkinsons Disease
<br>.Diabetes

<h2><b>Models with their accuracy</b></h2>

<br>Disease&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp               Model&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp              Accuracy
<br>Diabetes&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp	  Machine Learning Model&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp	  95.25%
<br>Parkinson's&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp     Machine Learning Model&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp	  97.44%
<br>Heart Disease&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp	  Machine Learning Model&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp     97.56%

<h2><b>Steps to run this application in your system</b></h2>

<br>1.Clone or download the repo.<br>
<br>2.Open command prompt in the downloaded folder.<br>
<br>3.Create a virtual environment<br>
<br>4.Install all the dependencies using this command in your command prompt:
        <br>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsppip install -r requirements.txt<br>
<br>5.Run the application using this command:
          <br>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp python app.py

