from flask import Flask, render_template, request
import pickle
import torch
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

# Define the LSTM-based text classifier
class LSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = self.fc(out[:, -1, :])
        return out

class NeuralNetworkClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetworkClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Load LSTM Model and Vectorizer
with open('models/lstm_classifier.pkl', 'rb') as f:
    lstm_classifier = pickle.load(f)

with open('models/lstm_vectorizer.pkl', 'rb') as f:
    lstm_vectorizer = pickle.load(f)

# Load Neural Network Model and Vectorizer
with open('models/nn_classifier.pkl', 'rb') as f:
    nn_classifier = pickle.load(f)

with open('models/nn_vectorizer.pkl', 'rb') as f:
    nn_vectorizer = pickle.load(f)

@app.route('/')
def home():
    print("Home page accessed")
    return render_template('index.html')

@app.route('/predict-lstm', methods=['POST'])
def predict_lstm():
    text = request.form['text']
    # Convert the input text to a numerical vector using the LSTM Vectorizer
    text_vectorized = lstm_vectorizer.transform([text])

    # Convert the vectorized text to a PyTorch tensor
    text_tensor = torch.tensor(text_vectorized.toarray(), dtype=torch.float32)

    # Make predictions using the LSTM model
    lstm_classifier.eval()
    with torch.no_grad():
        output = lstm_classifier(text_tensor)
        _, predicted = torch.max(output, 1)

    # Return the predicted sentiment (0 for negative, 1 for positive)
    predicted_sentiment = "positive" if predicted.item() == 1 else "negative"

    return render_template('index.html', prediction_lstm=predicted_sentiment)

@app.route('/predict-nn', methods=['POST'])
def predict_nn():
    text = request.form['text']
    # Convert the input text to a numerical vector using the Neural Network Vectorizer
    text_vectorized = nn_vectorizer.transform([text])

    # Convert the vectorized text to a PyTorch tensor
    text_tensor = torch.tensor(text_vectorized.toarray(), dtype=torch.float32)

    # Make predictions using the Neural Network model
    nn_classifier.eval()
    with torch.no_grad():
        output = nn_classifier(text_tensor)
        _, predicted = torch.max(output, 1)

    # Return the predicted sentiment (0 for negative, 1 for positive)
    predicted_sentiment = "positive" if predicted.item() == 1 else "negative"

    return render_template('index.html', prediction_nn=predicted_sentiment)

if __name__ == '__main__':
    app.run(debug=True)
