import pandas as pd
import joblib

path = "C:/Users/hadda/OneDrive/Documents/GitHub/Machine-learning-Elias-El-Haddad/Lab_ML/"

# Load the saved model and test samples
model = joblib.load(path + "best_dt_model.pkl")
test_samples = pd.read_csv(path + "test_samples.csv")

# Drop the target column from the test samples
X_test = test_samples.drop("cardio", axis=1)

# Make predictions and get probabilities
predictions = model.predict(X_test)
probs = model.predict_proba(X_test)

# Create a DataFrame with the predictions and probabilities
result_df = pd.DataFrame(probs, columns=["probability_class_0", "probability_class_1"])
result_df["prediction"] = predictions

# Export the results to a CSV file
result_df.to_csv(path + "prediction.csv", index=False)

print(result_df)
print(predictions)