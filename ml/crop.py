import pandas as pd
import os 
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer

MODEL_FILE = 'model.pkl'

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("Hidden/Crop_recommendation.csv")
    # print(housing.head())
    
    # housing['label_cat']=pd.cut(housing['label'], bins = [0.0,1.5,3.0,4.5,6.0, np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index , test_index in split.split(housing, housing['label']):
          strat_train_set = housing.loc[train_index]
          strat_test_set  = housing.loc[test_index]
          
    y_real = strat_test_set['label']
    
    strat_test_set.to_csv("compare.csv", index=False)
    print("Data for comparision is stored in compare.csv")
    strat_test_set.drop("label", axis = 1).to_csv("input.csv",index = False)
    print("Your test data is stored in input.csv")
    housing_labels = strat_train_set['label'].copy()
    housing_features = strat_train_set.drop('label', axis = 1)
    # print(housing_labels)
    # print(housing_features)
    
    model = RandomForestClassifier()
    print("Your model is fitting the value...")
    model.fit(housing_features, housing_labels)
    joblib.dump(model, MODEL_FILE)
    print("Model is trained. Congrats!")

else:
    model = joblib.load(MODEL_FILE)
    
 
    input_data = pd.DataFrame([
        {
    "N": 50,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.50,
    "rainfall": 202.93
        }]
    )
    prediction = model.predict(input_data)
    print("Interference is complete, result save to output.csv file. Enjoy!!")
    print("Recommended crop:", prediction[0])

    # ✅ Evaluate accuracy on test set
    data = pd.read_csv("compare.csv")
    X_test = data.drop("label", axis=1)
    y_real = data["label"]
    y_pred = model.predict(X_test)
    print("Model accuracy on test set:", accuracy_score(y_real, y_pred))