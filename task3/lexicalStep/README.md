# 📊 Sentiment Analysis - Lexical Models

## 🧠 Overview

In this project, we implemented two lexical-based sentiment analysis models:

* **Bing Liu Model**
* **SWN (Lexicon-Based) Model**

Both models rely on predefined dictionaries instead of machine learning techniques.

---

## ⚙️ Project Pipeline

1. Load dataset from CSV
2. Apply text preprocessing
3. Tokenize review text
4. Apply sentiment models:

   * Bing Model
   * SWN Lexicon Model
5. Generate predictions
6. Analyze results

---

## 🔵 Bing Model

The Bing model uses two dictionaries:

* `positive.txt`
* `negative.txt`

### How it works:

* Each positive word → `+1`
* Each negative word → `-1`
* Final score = sum of all words

### Output:

* Score > 0 → Positive
* Score < 0 → Negative
* Score = 0 → Neutral

### Advantages:

* Simple and fast
* Easy to implement

### Limitations:

* All words have equal weight
* Cannot capture sentiment intensity

---

## 🟢 SWN Lexicon Model

The SWN-inspired model uses the same dictionaries but applies a scoring approach.

### How it works:

* Each word is assigned a score:

  * Positive → `+1`
  * Negative → `-1`
* Final score = sum of word scores

### Output:

* Positive / Negative / Neutral (based on total score)

### Advantages:

* More flexible than Bing
* Can be extended to weighted scoring

### Limitations:

* Still dictionary-based
* Depends heavily on word coverage

---

## 🔍 Key Differences

| Feature     | Bing Model    | SWN Model   |
| ----------- | ------------- | ----------- |
| Approach    | Word counting | Score-based |
| Complexity  | Simple        | Moderate    |
| Flexibility | Low           | Higher      |
| Sensitivity | Limited       | Better      |

---

## 📈 Observations

* Many reviews are classified as **neutral** when words are not found in dictionaries
* Results improve with larger and richer lexicons
* Both models struggle with:

  * Negation (e.g., "not good")
  * Context understanding
  * Sarcasm

---

## ⚠️ Limitations

* No understanding of sentence structure
* No context awareness
* Performance depends on dictionary quality

---

## ✅ Conclusion

* The **Bing model** provides a simple baseline for sentiment analysis
* The **SWN model** improves on this by using a scoring approach
* Both models are useful for understanding basic sentiment analysis concepts
* However, they are limited compared to machine learning approaches

---

## 🚀 Future Improvements

* Use larger sentiment lexicons
* Add negation handling
* Apply machine learning models (Naive Bayes, Decision Tree)
* Use advanced embeddings (e.g., Word2Vec, GloVe)

---
