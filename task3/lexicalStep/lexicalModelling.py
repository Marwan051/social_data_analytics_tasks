import pandas as pd
from utils.preprocessing import clean_text
from models.bing_model import BingSentiment
from models.sentiwordnet_model import SWNSentiment
# load data
df = pd.read_csv("../../data/scored_data/cleaned_reviews_all_flags_scored.csv")


# preprocessing
df["tokens"] = df["review_body"].apply(clean_text)

# -------------------------------
# Bing Model 
# -------------------------------
bingModel = BingSentiment("positive.txt", "negative.txt")

df["bing_prediction"] = df["tokens"].apply(bingModel.predict)


# -------------------------------
# SWN Model
# -------------------------------
swn_model = SWNSentiment("positive.txt", "negative.txt")

df["swn_prediction"] = df["tokens"].apply(swn_model.predict)



# show results
print("\nSample Results of the Bing Model: >>>>>>>>>>>>>>>>>>>>>\n")
for i in range(5):
    print(df["review_body"].iloc[i])
    print("→", df["bing_prediction"].iloc[i])
    print("-" * 50)

print("\n\n\nSample Results of the SWN Model: >>>>>>>>>>>>>>>>>>>>>\n")
for i in range(5):
    print(df["review_body"].iloc[i])
    print("→", df["swn_prediction"].iloc[i])
    print("-" * 50)

# save
df.to_csv("output.csv", index=False)



# Analysis Prediction Count
print(f"\n\n{df["bing_prediction"].value_counts()}\n")
print(f"\n{df["swn_prediction"].value_counts()}\n\n")

# Diffrence Between the two Models
for i in range(5):
    print("Title:", df["review_title"].iloc[i])
    print("Bing:", df["bing_prediction"].iloc[i])
    print("SWN:", df["swn_prediction"].iloc[i])
    print("-" * 40)