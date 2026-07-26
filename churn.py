import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



DATASET_FILE = "customer_churn.csv"
MODEL_FILE = "churn_model.pkl"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "encoders.pkl"
PREDICTION_HISTORY_FILE = "prediction_history.csv"


def create_dataset():
    if os.path.exists(DATASET_FILE):
        print(f"Dataset '{DATASET_FILE}' already exists. Skipping creation.")
        return

    print(f"Generating synthetic dataset '{DATASET_FILE}'...")
    np.random.seed(42)
    num_samples = 1000

    genders = ["Male", "Female"]
    contract_types = ["Month-to-month", "One year", "Two year"]
    internet_services = ["DSL", "Fiber optic", "No"]
    payment_methods = [
        "Electronic check",
        "Mailed check",
        "Bank transfer",
        "Credit card",
    ]
    yes_no_options = ["Yes", "No"]

    ages = np.random.randint(18, 70, size=num_samples)
    gender_col = np.random.choice(genders, size=num_samples)
    tenures = np.random.randint(1, 72, size=num_samples)
    monthly_charges = np.round(
        np.random.uniform(20.0, 120.0, size=num_samples), 2
    )

    total_charges = np.round(
        monthly_charges * tenures + np.random.normal(0, 50, size=num_samples),
        2,
    )
    total_charges = np.maximum(
        total_charges, monthly_charges
    )  # Avoid total < monthly

    contracts = np.random.choice(
        contract_types, size=num_samples, p=[0.5, 0.3, 0.2]
    )
    internet = np.random.choice(
        internet_services, size=num_samples, p=[0.4, 0.4, 0.2]
    )
    payment = np.random.choice(payment_methods, size=num_samples)
    tech_supp = np.random.choice(yes_no_options, size=num_samples)
    online_sec = np.random.choice(yes_no_options, size=num_samples)

  
    churn_prob = (
        0.3
        + (ages > 50) * 0.15
        + (contracts == "Month-to-month") * 0.25
        - (tenures > 24) * 0.20
        + (monthly_charges > 80) * 0.10
        - (tech_supp == "Yes") * 0.10
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "Age": ages,
            "Gender": gender_col,
            "Tenure": tenures,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contracts,
            "InternetService": internet,
            "PaymentMethod": payment,
            "TechSupport": tech_supp,
            "OnlineSecurity": online_sec,
            "Churn": churn,
        }
    )

    df.to_csv(DATASET_FILE, index=False)
    print("Dataset created successfully.\n")


def load_data():
    """Loads dataset from CSV and prints basic Exploratory Data Analysis (EDA) info."""
    print("--- Loading Dataset & Exploratory Data Analysis (EDA) ---")
    try:
        df = pd.read_csv(DATASET_FILE)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    print("\n1. First 5 records:")
    print(df.head())

    print("\n2. Last 5 records:")
    print(df.tail())

    print(f"\n3. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n4. Data Types:")
    print(df.dtypes)

    print("\n5. Missing Values:")
    print(df.isnull().sum())

    print(f"\n6. Duplicate Values: {df.duplicated().sum()}")

    print("\n7. Statistical Summary:")
    print(df.describe())
    print("-" * 50)

    return df


def preprocess_data(df):
    """Handles missing values, label encoding, scaling, and train-test splitting."""
    print("\n--- Data Preprocessing ---")
    df = df.copy()

    # Handle missing values if any
    if df.isnull().sum().sum() > 0:
        df.fillna(df.median(numeric_only=True), inplace=True)

    categorical_cols = [
        "Gender",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
    ]
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to maintain column names for feature importances
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Save scaler and encoders for future terminal predictions
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(encoders, ENCODERS_FILE)

    print("Data preprocessed, encoded, and scaled successfully.")
    return X_train_scaled, X_test_scaled, y_train, y_test, encoders, scaler


def train_model(X_train, y_train):
    """Trains a Random Forest Classifier or loads an existing trained model."""
    if os.path.exists(MODEL_FILE):
        print(f"\nLoading existing trained model from '{MODEL_FILE}'...")
        model = joblib.load(MODEL_FILE)
    else:
        print("\nTraining Random Forest Classifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
        print(f"Model trained and saved to '{MODEL_FILE}'.")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates the model using various classification metrics."""
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision:       {prec:.4f}")
    print(f"Recall:          {rec:.4f}")
    print(f"F1 Score:        {f1:.4f}")
    print(f"ROC-AUC Score:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def visualize_data(df, model, feature_names):
    """Generates all required visual plots using Matplotlib."""
    print("\nDisplaying Exploratory & Model Visualizations...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle("Customer Churn Data & Model Analysis", fontsize=16)

    # 1. Churn Distribution
    churn_counts = df["Churn"].value_counts()
    axes[0, 0].bar(
        ["Stayed (0)", "Churned (1)"], churn_counts.values, color=["skyblue", "salmon"]
    )
    axes[0, 0].set_title("Churn Distribution")
    axes[0, 0].set_ylabel("Count")

    # 2. Age Distribution
    axes[0, 1].hist(df["Age"], bins=15, color="teal", edgecolor="black")
    axes[0, 1].set_title("Age Distribution")
    axes[0, 1].set_xlabel("Age")
    axes[0, 1].set_ylabel("Frequency")

    # 3. Monthly Charges Distribution
    axes[1, 0].hist(
        df["MonthlyCharges"], bins=15, color="mediumpurple", edgecolor="black"
    )
    axes[1, 0].set_title("Monthly Charges Distribution")
    axes[1, 0].set_xlabel("Monthly Charges ($)")
    axes[1, 0].set_ylabel("Frequency")

    # 4. Contract Type Distribution
    contract_counts = df["Contract"].value_counts()
    axes[1, 1].bar(
        contract_counts.index, contract_counts.values, color="coral"
    )
    axes[1, 1].set_title("Contract Type Distribution")
    axes[1, 1].set_ylabel("Count")

    # 5. Correlation Heatmap
    # Convert categorical text to numeric temporary copies for correlation
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    corr = df_encoded.corr()
    im = axes[2, 0].imshow(corr, cmap="coolwarm", interpolation="nearest")
    axes[2, 0].set_title("Correlation Heatmap")
    axes[2, 0].set_xticks(range(len(corr.columns)))
    axes[2, 0].set_yticks(range(len(corr.columns)))
    axes[2, 0].set_xticklabels(corr.columns, rotation=90, fontsize=8)
    axes[2, 0].set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=axes[2, 0])

    # 6. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)
    axes[2, 1].barh(
        range(len(indices)), importances[indices], color="forestgreen"
    )
    axes[2, 1].set_yticks(range(len(indices)))
    axes[2, 1].set_yticklabels([feature_names[i] for i in indices])
    axes[2, 1].set_title("Feature Importance (Random Forest)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_user_choice(prompt, valid_options):
    """Helper function to prompt user terminal inputs with validation."""
    while True:
        value = input(prompt).strip()
        matching = [
            opt
            for opt in valid_options
            if opt.lower() == value.lower()
        ]
        if matching:
            return matching[0]
        print(f"Invalid input! Please choose from: {', '.join(valid_options)}")


def predict_customer(model, scaler, encoders):
    """Interactively prompts for customer details, runs prediction, and logs history."""
    print("\n--- Predict Customer Churn (Interactive Mode) ---")

    try:
        age = float(input("Enter Age (e.g., 45): "))
        gender = get_user_choice("Enter Gender (Male/Female): ", ["Male", "Female"])
        tenure = float(input("Enter Tenure in months (e.g., 12): "))
        monthly_charges = float(input("Enter Monthly Charges ($) (e.g., 75.50): "))
        total_charges = float(input("Enter Total Charges ($) (e.g., 906.00): "))

        contract = get_user_choice(
            "Enter Contract Type (Month-to-month/One year/Two year): ",
            ["Month-to-month", "One year", "Two year"],
        )
        internet = get_user_choice(
            "Enter Internet Service (DSL/Fiber optic/No): ",
            ["DSL", "Fiber optic", "No"],
        )
        payment = get_user_choice(
            "Enter Payment Method (Electronic check/Mailed check/Bank transfer/Credit card): ",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer",
                "Credit card",
            ],
        )
        tech_support = get_user_choice(
            "Enter Tech Support (Yes/No): ", ["Yes", "No"]
        )
        online_security = get_user_choice(
            "Enter Online Security (Yes/No): ", ["Yes", "No"]
        )

        user_raw = {
            "Age": age,
            "Gender": gender,
            "Tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet,
            "PaymentMethod": payment,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
        }

        user_encoded = user_raw.copy()

        # Apply encoders
        for col, le in encoders.items():
            user_encoded[col] = le.transform([user_raw[col]])[0]

        # Convert input dictionary into DataFrame matching original column structure
        user_df = pd.DataFrame([user_encoded])
        user_scaled = scaler.transform(user_df)

        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0][1]

        result = "Customer will Leave" if prediction == 1 else "Customer will Stay"

        print("\n================ PREDICTION RESULT ================")
        print(f"Status:      {result}")
        print(f"Probability of Churning: {probability * 100:.2f}%")
        print("===================================================")

        # Log to prediction_history.csv
        history_record = user_raw.copy()
        history_record["Prediction"] = result
        history_record["Churn_Probability"] = round(probability, 4)

        history_df = pd.DataFrame([history_record])

        if not os.path.exists(PREDICTION_HISTORY_FILE):
            history_df.to_csv(PREDICTION_HISTORY_FILE, index=False)
        else:
            history_df.to_csv(
                PREDICTION_HISTORY_FILE, mode="a", header=False, index=False
            )

        print(f"Prediction appended to '{PREDICTION_HISTORY_FILE}'.")

    except ValueError as e:
        print(f"Input Error: Please enter valid numbers. ({e})")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


def main():
    """Main execution flow of the project."""
    print("==================================================")
    print("      Customer Churn Prediction System            ")
    print("==================================================\n")

    # Step 1: Dataset Creation
    create_dataset()

    # Step 2: Load Data & EDA
    df = load_data()
    if df is None:
        return

    # Step 3: Preprocess Data
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_data(df)

    # Step 4: Model Training / Loading
    model = train_model(X_train, y_train)

    # Step 5: Evaluate Model
    evaluate_model(model, X_test, y_test)

    # Step 6: Data & Feature Visualizations
    visualize_data(df, model, X_train.columns)

    # Step 7: Interactive Terminal Prediction
    while True:
        predict_customer(model, scaler, encoders)
        again = input("\nPredict for another customer? (yes/no): ").strip().lower()
        if again != "yes":
            print("\nThank you for using the Customer Churn Prediction system!")
            break


if __name__ == "__main__":
    main()