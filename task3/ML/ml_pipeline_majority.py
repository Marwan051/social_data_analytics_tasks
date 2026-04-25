import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TARGET_COLUMN = "final_sentiment"

schemes = ["all_flags", "no_lemma_no_stopwords", "no_special_no_lowercase"]

models = {
    "Naive Bayes (BoW)": MultinomialNB(),
    "Naive Bayes (GloVe)": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}


def run_ml_pipeline_majority():
    print("Initiating Majority-Vote Machine Learning Pipeline...\n")

    for scheme in schemes:
        print("=" * 50)
        print(f"Evaluation Scheme: {scheme.upper()}")
        print("=" * 50)

        labels_path = f"../../data/scored_data/cleaned_reviews_{scheme}_majority.csv"

        bow_path = f"../text representation/bow_{scheme}.csv"
        glove_path = f"../text representation/glove_{scheme}.csv"

        try:
            # 1. Load the Data
            df_labels = pd.read_csv(labels_path)
            y = df_labels[TARGET_COLUMN]
            print(y.value_counts())
            X_bow = pd.read_csv(bow_path)
            X_glove = pd.read_csv(glove_path)

            # Ensure the text rep files only contain numeric features
            # AND drop the original scores to prevent Data Leakage!
            leakage_columns = [
                TARGET_COLUMN,
                "rating",
                "score_gemini",
                "score_groq",
                "final_score",
            ]

            for col in leakage_columns:
                if col in X_bow.columns:
                    X_bow = X_bow.drop(columns=[col])
                if col in X_glove.columns:
                    X_glove = X_glove.drop(columns=[col])

            X_bow = X_bow.select_dtypes(include=["number"])
            X_glove = X_glove.select_dtypes(include=["number"])

            # 2. Bag-of-Words Evaluation
            print("\n--- Text Rep: Bag-of-Words ---")
            X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
                X_bow, y, test_size=0.2, random_state=42
            )

            # Train & Test Naive Bayes on BoW
            nb_bow = models["Naive Bayes (BoW)"]
            nb_bow.fit(X_train_b, y_train_b)
            nb_bow_acc = accuracy_score(y_test_b, nb_bow.predict(X_test_b))
            print(f"Naive Bayes Accuracy:   {nb_bow_acc:.4f}")

            # Train & Test Decision Tree on BoW
            dt_bow = models["Decision Tree"]
            dt_bow.fit(X_train_b, y_train_b)
            dt_bow_acc = accuracy_score(y_test_b, dt_bow.predict(X_test_b))
            print(f"Decision Tree Accuracy: {dt_bow_acc:.4f}")

            # 3. GloVe Evaluation
            print("\n--- Text Rep: GloVe ---")
            X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
                X_glove, y, test_size=0.2, random_state=42
            )

            # Train & Test Naive Bayes on GloVe
            nb_glove = models["Naive Bayes (GloVe)"]
            nb_glove.fit(X_train_g, y_train_g)
            nb_glove_acc = accuracy_score(y_test_g, nb_glove.predict(X_test_g))
            print(f"Naive Bayes Accuracy:   {nb_glove_acc:.4f}")

            # Train & Test Decision Tree on GloVe
            dt_glove = models["Decision Tree"]
            dt_glove.fit(X_train_g, y_train_g)
            dt_glove_acc = accuracy_score(y_test_g, dt_glove.predict(X_test_g))
            print(f"Decision Tree Accuracy: {dt_glove_acc:.4f}\n")

        except FileNotFoundError as e:
            print(f"!! Error: Could not find file. {e}")
            print("Skipping to next scheme...\n")
        except KeyError as e:
            print(f"!! Error: Column {e} not found in the labels file.")
            print(
                f"Please check the exact column name in '{labels_path}' and update TARGET_COLUMN at the top of the script.\n"
            )


if __name__ == "__main__":
    run_ml_pipeline_majority()
