AI-Based Crop Recommendation
🌱 Project Overview

This project is an AI-based crop recommendation system.
It uses a machine learning model (Random Forest Classifier) trained on soil and environmental parameters to predict the most suitable crop for given conditions.

The model is trained once, saved as a .pkl file, and can later be loaded for fast predictions without retraining.

📊 Dataset Information (Crop_recommendation.csv)

The dataset contains multiple rows of crop conditions and the corresponding crop label.

Column	Description
N	Nitrogen content in soil (mg/kg)
P	Phosphorus content in soil (mg/kg)
K	Potassium content in soil (mg/kg)
temperature	Average temperature of the region (°C)
humidity	Relative humidity (%)
ph	pH value of the soil
rainfall	Average rainfall (mm)
label	Target column → crop name (e.g., rice, maize, orange, grapes, etc.)
⚙️ How the Code Works

Import Libraries

pandas, numpy → data handling

joblib → saving/loading model

sklearn → training ML model

Model Training

Dataset is read from Crop_recommendation.csv

Features (N, P, K, temperature, humidity, ph, rainfall) are separated from target (label)

Dataset is split into training (80%) and testing (20%) using stratified sampling

A RandomForestClassifier is trained on the data

Model Saving

The trained model is saved as new_model.pkl

Model Loading

If new_model.pkl already exists, it loads the model directly (no retraining needed)

Prediction

The model predicts the most suitable crop for a given set of soil/environment inputs

🖥️ Example Code
Training and Saving Model
if not os.path.exists("new_model.pkl"):
    housing = pd.read_csv("Crop_recommendation.csv")
    X = housing.drop("label", axis=1)
    y = housing["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "new_model.pkl")
    print("✅ Model trained and saved as new_model.pkl")

Loading Model & Making Prediction
else:
    model = joblib.load("new_model.pkl")
    print("Model loaded from new_model.pkl")

    new_data = pd.DataFrame([{
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 6.87,
        "humidity": 92.00,
        "ph": 7.50,
        "rainfall": 20.93
    }])

    prediction = model.predict(new_data)
    print("🌱 Recommended crop:", prediction[0])

📌 Example Output
✅ Model trained and saved as new_model.pkl
Model loaded from new_model.pkl
🌱 Recommended crop: rice