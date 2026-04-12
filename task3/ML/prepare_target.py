import pandas as pd
from collections import Counter

# 1. Convert numeric scores to sentiment text
def convert_to_sentiment(score):
    try:
        score = float(score)
        if score >= 4:
            return 'positive'
        elif score <= 2:
            return 'negative'
        else:
            return 'neutral'
    except:
        return 'neutral'

# 2. Find the majority vote for sentiment
def get_majority_sentiment(row):
    r_sent = row['rating_sentiment']
    gemini_sent = row['gemini_sentiment']
    groq_sent = row['groq_sentiment']
    
    votes = [r_sent, gemini_sent, groq_sent]
    vote_counts = Counter(votes)
    top_vote, top_count = vote_counts.most_common(1)[0]
    
    if top_count == 1:
        return r_sent
    return top_vote

# 3. Find the majority vote for scores
def get_majority_score(row):
    try:
        r_score = float(row['rating'])
    except:
        r_score = 3.0 
        
    try:
        g_score = float(row['score_gemini'])
    except:
        g_score = 3.0
        
    try:
        q_score = float(row['score_groq'])
    except:
        q_score = 3.0
        
    votes = [r_score, g_score, q_score]
    vote_counts = Counter(votes)
    
    top_vote, top_count = vote_counts.most_common(1)[0]
    
    # Tie-breaker: trust the user's rating
    if top_count == 1:
        return r_score
        
    return top_vote

def process_target_columns():
    schemes = [
        "all_flags",
        "no_lemma_no_stopwords",
        "no_speical_no_lowercase"
    ]
    
    print("Pls wait...\n")
    
    for scheme in schemes:
        input_path = f"../../data/scored_data/cleaned_reviews_{scheme}_scored.csv"
        output_path = f"../../data/scored_data/cleaned_reviews_{scheme}_majority.csv"
        
        try:
            df = pd.read_csv(input_path)
            
            # --- CREATE THE TEXT COLUMNS ---
            df['rating_sentiment'] = df['rating'].apply(convert_to_sentiment)
            df['gemini_sentiment'] = df['score_gemini'].apply(convert_to_sentiment)
            df['groq_sentiment'] = df['score_groq'].apply(convert_to_sentiment)
            
            # --- CALCULATE MAJORITIES ---
            df['final_sentiment'] = df.apply(get_majority_sentiment, axis=1) 
            df['final_score'] = df.apply(get_majority_score, axis=1)         
                       
            df.to_csv(output_path, index=False)
            print(f"Successfully created new file: cleaned_reviews_{scheme}_majority.csv")
            
        except FileNotFoundError:
            print(f"!! Error: Could not find {input_path}")
        except KeyError as e:
            print(f"!! Error: Missing column {e} in {scheme}. Check spelling!")

if __name__ == "__main__":
    process_target_columns()