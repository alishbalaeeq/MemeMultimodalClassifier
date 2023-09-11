import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import HashingVectorizer
from skimage.io import imread
from skimage.transform import resize
from natsort import natsorted
from PIL import ImageFile
import pickle

# Set image loading options
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Extract data from the ZIP file
zip_path = '/content/drive/MyDrive/archive.zip'
extract_path = '/content/sample_data'
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load additional image features from pickle
image_features = pickle.load(open('More_Features.pkl', 'rb'))

# Load the cleaned CSV file
df = pd.read_csv('clean.csv')

# Encode text data using HashingVectorizer
vectorizer = HashingVectorizer(n_features=50)
encoded_text = vectorizer.fit_transform(df['text_corrected']).toarray()

# Flatten encoded text data
flat_encoded_text = [text.flatten() for text in encoded_text]

# Create a DataFrame for combined features
features = pd.DataFrame({'Image': image_features, 'Text': flat_encoded_text})

# Define labels for sentiment
sentiment_labels = df.overall_sentiment.map({'negative': 0, 'neutral': 1, 'positive': 2, 'very_positive': 2, 'very_negative': 0})

# Define labels for humor, sarcasm, offensive, and motivational
humor_labels = df['humour'].map({'funny': 1, 'very_funny': 1, 'not_funny': 0, 'hilarious': 1})
sarcasm_labels = df['sarcasm'].map({'general': 1, 'not_sarcastic': 0, 'twisted_meaning': 1, 'very_twisted': 1})
offensive_labels = df['offensive'].map({'not_offensive': 0, 'slight': 1, 'very_offensive': 1, 'hateful_offensive': 1})
motivational_labels = df['motivational'].map({'not_motivational': 0, 'motivational': 1})

# Create train-test splits for sentiment classification
ImageX_train, ImageX_test, ImageY_train, ImageY_test = train_test_split(features['Image'], sentiment_labels, test_size=0.33, random_state=42)
TextX_train, TextX_test, TextY_train, TextY_test = train_test_split(features['Text'], sentiment_labels, test_size=0.33, random_state=42)

# Convert data to PyTorch tensors
ImageX_trainTensor = torch.Tensor(list(ImageX_train))
ImageY_trainTensor = torch.Tensor(ImageY_train.values)
ImageX_testTensor = torch.Tensor(list(ImageX_test))
ImageY_testTensor = torch.Tensor(ImageY_test.values)

TextX_trainTensor = torch.Tensor(list(TextX_train))
TextY_trainTensor = torch.Tensor(list(TextY_train))
TextX_testTensor = torch.Tensor(list(TextX_test))
TextY_testTensor = torch.Tensor(TextY_test.values)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()

        # Image Input Layer
        self.ILayer1 = nn.Linear(500*500, 500)
        self.ILayer2 = nn.Linear(500, 400)
        self.ILayer3 = nn.Linear(400, 400)
        self.ILayer4 = nn.Linear(400, 400)
        self.IOutPut = nn.Linear(400, 120)

        # Text Input Layer
        self.TLayer1 = nn.Linear(50, 500)
        self.TLayer2 = nn.Linear(500, 400)
        self.TLayer3 = nn.Linear(400, 400)
        self.TLayer4 = nn.Linear(400, 400)
        self.TLayer5 = nn.Linear(400, 400)
        self.TOutPut = nn.Linear(400, 120)

        # Combining Image and Text Layers
        self.together = nn.Linear(240, 500)
        self.together1 = nn.Linear(500, 500)
        self.together2 = nn.Linear(500, 500)
        self.together3 = nn.Linear(500, 500)
        self.together4 = nn.Linear(500, 250)
        self.outputTogether = nn.Linear(250, 80)

        # Humor Classification Layer
        self.humour = nn.Linear(80, 20)
        self.hout = nn.Linear(20, 2)

        # Sarcasm Classification Layer
        self.sarcasm = nn.Linear(80, 20)
        self.sout = nn.Linear(20, 2)

        # Offensive Classification Layer
        self.offensive = nn.Linear(80, 20)
        self.Oout = nn.Linear(20, 2)

        # Motivational Classification Layer
        self.motivational = nn.Linear(80, 20)
        self.Mout = nn.Linear(20, 2)

    def forward(self, image, text):
        # Forward pass for image features
        x = self.ILayer1(image)
        x = nn.ReLU()(x)
        x = self.ILayer2(x)
        x = nn.ReLU()(x)
        x = self.ILayer3(x)
        x = nn.ReLU()(x)
        x = self.ILayer4(x)
        x = nn.ReLU()(x)
        x = self.IOutPut(x)

        # Forward pass for text features
        y = self.TLayer1(text)
        y = nn.ReLU()(y)
        y = self.TLayer2(y)
        y = nn.ReLU()(y)
        y = self.TLayer3(y)
        y = nn.ReLU()(y)
        y = self.TLayer4(y)
        y = nn.ReLU()(y)
        y = self.TLayer5(y)
        y = nn.ReLU()(y)
        y = self.TOutPut(y)

        # Concatenate image and text features
        concatenate = torch.cat((x, y), dim=1)

        # Forward pass for combined features
        concatenate = self.together(concatenate)
        concatenate = nn.ReLU()(concatenate)
        concatenate = self.together1(concatenate)
        concatenate = nn.ReLU()(concatenate)
        concatenate = self.together2(concatenate)
        concatenate = nn.ReLU()(concatenate)
        concatenate = self.together3(concatenate)
        concatenate = nn.ReLU()(concatenate)
        concatenate = self.together4(concatenate)
        concatenate = nn.ReLU()(concatenate)
        output = self.outputTogether(concatenate)

        # Humor Classification
        h = self.humour(output)
        h = nn.ReLU()(h)
        h = self.hout(h)

        # Sarcasm Classification
        s = self.sarcasm(output)
        s = nn.ReLU()(s)
        s = self.sout(s)

        # Offensive Classification
        o = self.offensive(output)
        o = nn.ReLU()(o)
        o = self.Oout(o)

        # Motivational Classification
        m = self.motivational(output)
        m = nn.ReLU()(m)
        m = self.Mout(m)

        return h, s, o, m

# Create an instance of the neural network
neural = ANN()

# Define a function to get labels for humor, sarcasm, offensive, and motivational
def Label_idx(idx):
    array = [
        humor_labels[idx],
        sarcasm_labels[idx],
        offensive_labels[idx],
        motivational_labels[idx]
    ]
    return array

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(neural.parameters(), lr=0.001, momentum=0.9)

# Training loop
for epoch in range(1):
    prediction = []
    for i in range(len(ImageX_trainTensor)):
        optimizer.zero_grad()
        y_pred = neural(ImageX_trainTensor[i].cuda(), TextX_trainTensor[i].cuda())
        labels = Label_idx(i)
        loss = 0
        for j in range(len(y_pred)):
            loss += criterion(y_pred[j].unsqueeze(0), torch.tensor(labels[j]).long().unsqueeze(0).cuda())
        loss.backward()
        optimizer.step()
        if i % 2000 == 1999:
            print(f'Epoch {epoch + 1} loss: {loss.item() / 2000:.3f}')

# Evaluation
prediction = []
Labels = []
for i in range(len(ImageX_testTensor)):
    y_pred = neural(ImageX_testTensor[i].cuda(), TextX_testTensor[i].cuda())
    labels = Label_idx(i)
    for j in range(len(y_pred)):
        prediction.append(int(torch.argmax(y_pred[j])))
        Labels.append(labels[j])

print("Accuracy for Testing: ", accuracy_score(Labels, prediction) * 100)
print("Report for Testing: \n", classification_report(Labels, prediction))
print("F1 Score for Testing: ", f1_score(Labels, prediction, average='macro') * 100)

# Save the model
torch.save(neural, 'task2')
