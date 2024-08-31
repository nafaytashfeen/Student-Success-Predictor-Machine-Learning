# Student Success Predictor with Machine Learning

## This is a short project in which a machine learning model is trained to predict whether a student will pass or fail a course based on certaain factors

How it works:

Data Collection: A dataset is needed to train the model. The dataset used in this project contains various features related to student performance, such as study hours per week, attendance rate, previous grades, participation in extracurricular activities, and parent education level.

Model Training: A RandomForestClassifier is employed to train the model. The dataset is split into training and testing sets, with 80% of the data used for training. The model is trained using the specified predictors to classify whether a student will pass or fail.

Accuracy Testing: After the model has been trained, it can be tested using the test dataset and its accuracy score can be determined

User Interaction: The user can call the function <make_prediction> and test the model for themselves


How to use:

1. Clone the repository
Type this in the terminal:
`git clone https://github.com/nafaytashfeen/Student-Success-Predictor-Machine-Learning.git`

2. Navigate to the project directory
Type this in the terminal:
`cd Student Success Prediction`

3. Install required dependancies
These are found in the file requirements.txt. Type this in the terminal:
`pip install -r requirements.txt`

4. Test the model
Call the function <make_prediction> and test the model for yourself. Run the python file by typing this in the terminal: `python main.py`


