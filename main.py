import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

student_perf = pd.read_csv('student_performance_prediction.csv', index_col = 0)
student_perf = student_perf.dropna()
print(student_perf)

# Categorizing some of the data as Integer, (Participation in Extracurricular, 
# Parent Education, Passed/Failed)
student_perf["Participation in Extracurricular"] = student_perf['Participation in Extracurricular Activities'].astype('category').cat.codes
student_perf["Parent Education"] = student_perf['Parent Education Level'].astype('category').cat.codes
student_perf['Passed_failed'] = student_perf['Passed'].astype('category').cat.codes

# Create the model and begin training it

model = RandomForestClassifier(n_estimators = 50, min_samples_split=10, random_state=1)
train = student_perf[student_perf.index.str[1:].astype(int) < 32000]
test = student_perf[student_perf.index.str[1:].astype(int) >= 32000]
# Use 80% of the data for training, use the remaining for testing
predictors = ['Study Hours per Week', "Attendance Rate", "Previous Grades", "Participation in Extracurricular", "Parent Education"]
# The columns to use for predicting the data
model.fit(train[predictors], train['Passed_failed'])
# train the model
preds = model.predict(test[predictors])
# get the predictions

# Now to test the accuracy of the model

accuracy = accuracy_score(test['Passed_failed'], preds)
# Compute accuracy by checking actual with predictions

def make_prediction(study_hours: int, attendance_rate: float, previous_grade: float, 
                    extracurricular: bool, parent_education: str) -> str:
    """
    This function uses a model to predict whether a student will pass or fail a course 
    given the parameters. 

    The paremeters <attendance_rate> and <previous_grade> should be a float between 
    0 and 100 (inclusive) 0 <= x <= 100
    
    The parameter <extracurricular> is True if the student has extracurricular
    activities, and False otherwise.

    The parameter <parent_education> should be one of the following strings
    (High School, Bachelor, Master, Associate, Doctorate)
    """
    # Turn extracurricular into an int
    if extracurricular is False:
        extracurricular = 0
    
    else:
        extracurricular = 1

    # Turn parent_education into an int
    if parent_education == "Associate":
        parent_education = 0
    
    elif parent_education == "Bachelor":
        parent_education = 1

    elif parent_education == "Doctorate":
        parent_education = 2
    
    elif parent_education == "High School":
        parent_education = 3
    
    else:
        parent_education = 4

    input_data = pd.DataFrame({
        'Study Hours per Week': [study_hours],
        'Attendance Rate': [attendance_rate],
        'Previous Grades': [previous_grade],
        'Participation in Extracurricular': [extracurricular],
        'Parent Education': [parent_education]
    })

    # Make a prediction
    prediction = model.predict(input_data)

    # Interpret the result
    result = "Pass" if prediction[0] == 1 else "Fail"
    return result

# Example use
print(make_prediction(20, 78.9, 87, True, "Bachelor"))