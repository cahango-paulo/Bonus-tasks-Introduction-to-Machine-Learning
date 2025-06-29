import numpy as np
import re
from collections import Counter

class TopicClassifier:
    def __init__(self, num_topics):
        self.vocab = None
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.num_topics = num_topics
        self.weights = None
        self.bias = None
    
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def build_vocab(self, sentences, max_size=1000):
        word_counts = Counter()
        for sentence in sentences:
            tokens = self.preprocess(sentence)
            word_counts.update(tokens)
        common_words = word_counts.most_common(max_size)
        self.vocab = [word for word, _ in common_words]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
    
    def encode_sentence(self, sentence):
        tokens = self.preprocess(sentence)
        vector = np.zeros(len(self.vocab), dtype=np.float32)
        for token in tokens:
            if token in self.word_to_idx:
                idx = self.word_to_idx[token]
                vector[idx] += 1
        if len(tokens) > 0:
            vector /= len(tokens)
        return vector
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def train(self, sentences, labels, learning_rate=0.01, epochs=100):
        self.build_vocab(sentences)
        X = np.array([self.encode_sentence(s) for s in sentences])
        y = np.array(labels)
        n_features = X.shape[1]
        self.weights = np.random.randn(n_features, self.num_topics) * 0.01
        self.bias = np.zeros((1, self.num_topics))
        
        for epoch in range(epochs):
            z = np.dot(X, self.weights) + self.bias
            probs = self.softmax(z)
            m = X.shape[0]
            one_hot_y = np.eye(self.num_topics)[y]
            dz = probs - one_hot_y
            dw = np.dot(X.T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            if epoch % 10 == 0:
                loss = -np.mean(np.log(probs[np.arange(m), y]))
                predictions = np.argmax(probs, axis=1)
                accuracy = np.mean(predictions == y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
    
    def predict(self, sentence):
        if self.weights is None:
            raise Exception("Model is not trained yet!")
        x = self.encode_sentence(sentence)
        z = np.dot(x, self.weights) + self.bias
        probs = self.softmax(z[np.newaxis, :])
        return np.argmax(probs)


# Training data
train_sentences = [
    "machine learning is interesting",
    "neural networks recognize patterns",
    "the weather is beautiful today",
    "it is raining heavily",
    "the movie won an oscar",
    "the actors performed excellently"
]

train_labels = [0, 0, 1, 1, 2, 2]

# Initialize and train classifier
classifier = TopicClassifier(num_topics=3)
classifier.train(train_sentences, train_labels, epochs=100, learning_rate=0.1)

# Generate 100 test sentences
base_test_sentences = [
    "deep learning is used in science",
    "a thunderstorm is expected today",
    "the director received an award",
    "tomorrow's weather forecast",
    "scientific research involves models",
    "raining conditions are worsening",
    "actor performance was stunning",
    "temperature is dropping rapidly",
    "climate change affects the planet",
    "cinema continues to evolve"
]
test_sentences = base_test_sentences * 10  # Repeat 10 times to get 100

topic_names = {0: "Science", 1: "Weather", 2: "Film"}
results = [(sentence, topic_names[classifier.predict(sentence)]) for sentence in test_sentences[:10]]
print(results)

