class BingSentiment:

    def __init__(self, pos_path, neg_path):
        self.positive_words = self.load_words(pos_path)
        self.negative_words = self.load_words(neg_path)
        self.negations = {"not", "no", "never", "none", "n't"}

    def load_words(self, path):
        with open(path, "r") as f:
            return set(word.strip() for word in f.readlines())

    def predict(self, tokens):
        score = 0
        negate = False

        for word in tokens:

            if word in self.negations:
                negate = True
                continue

            if word in self.positive_words:
                score += -1 if negate else 1
                negate = False

            elif word in self.negative_words:
                score += 1 if negate else -1
                negate = False

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"