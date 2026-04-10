class SWNSentiment:

    def __init__(self, pos_path, neg_path):
        self.positive_words = self.load_words(pos_path)
        self.negative_words = self.load_words(neg_path)

    def load_words(self, path):
        with open(path, "r") as f:
            return set(word.strip() for word in f.readlines())

    def get_score(self, word):
        if word in self.positive_words:
            return 2   
        elif word in self.negative_words:
            return -1
        else:
            return 0

    def predict(self, tokens):
        score = sum(self.get_score(word) for word in tokens)

        if score > 0:
            return "positive"
        elif score < 0:
            return "negative"
        else:
            return "neutral"