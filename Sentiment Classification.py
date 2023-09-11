import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage import filters, feature, io, transform
from scipy.ndimage import rotate
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report
from natsort import natsorted
from PIL import ImageFile
from torchvision import transforms

# Set the seed for reproducibility
torch.manual_seed(0)

# Mount Google Drive if using Colab
from google.colab import drive
drive.mount('/content/drive')

# Define paths
zip_path = '/content/drive/MyDrive/archive.zip'
extract_path = '/content/sample_data'
image_path = '/content/sample_data/memotion_dataset_7k/images/'
labels_path = '/content/clean.csv'

# Extract the dataset
import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Load labels
df = pd.read_csv(labels_path)
df.drop(labels='Unnamed: 0', axis=1, inplace=True)

# Handle missing text values
nan_values = df[df['text_corrected'].isna()]
replace = df['text_ocr'][119]
df['text_corrected'][119] = replace
nan_values = nan_values[1:]

# Remove images corresponding to missing text values
for i in range(len(nan_values)):
    image = nan_values.iloc[i]
    address = image['image_name']
    os.remove(os.path.join(image_path, address))
    df.drop(df[df['image_name'] == address].index, inplace=True)

# Data augmentation functions
def data_augmentation_vertical_flip(negative, index, path):
    global df
    df_index = 0
    augment_images = negative['image_name']
    
    for i in augment_images:
        image_path = os.path.join(path, i)
        y, ext = i.split(".")
        
        image = io.imread(image_path)
        x = negative.iloc[df_index].copy()
        x['image_name'] = 'image_{}.{}'.format(index, ext)
        
        vertical_flip = np.flipud(image)
        vertical_flip = np.ascontiguousarray(vertical_flip)
        
        plt.imsave(os.path.join(path, 'image_{}.{}'.format(index, ext)), vertical_flip)
        
        index += 1
        df_index += 1
        df = df.append(x, ignore_index=True)

def data_augmentation_horizontal_flip(negative, index, path):
    global df
    df_index = 0
    augment_images = negative['image_name']
    
    for i in augment_images:
        image_path = os.path.join(path, i)
        y, ext = i.split(".")
        
        image = io.imread(image_path)
        x = negative.iloc[df_index].copy()
        x['image_name'] = 'image_{}.{}'.format(index, ext)
        
        horizontal_flip = np.fliplr(image)
        horizontal_flip = np.ascontiguousarray(horizontal_flip)
        
        plt.imsave(os.path.join(path, 'image_{}.{}'.format(index, ext)), horizontal_flip)
        
        index += 1
        df_index += 1
        df = df.append(x, ignore_index=True)

def data_augmentation_rotate(negative, index, path):
    global df
    df_index = 0
    augment_images = negative['image_name']
    
    for i in augment_images:
        image_path = os.path.join(path, i)
        y, ext = i.split(".")
        
        image = io.imread(image_path)
        rotated = rotate(image, 25)
        x = negative.iloc[df_index].copy()
        x['image_name'] = 'image_{}.{}'.format(index, ext)
        
        plt.imsave(os.path.join(path, 'image_{}.{}'.format(index, ext)), rotated)
        
        index += 1
        df_index += 1
        df = df.append(x, ignore_index=True)

# Perform data augmentation
negative = df.loc[df['overall_sentiment'] == 'negative'].copy()
very_negative = df.loc[df['overall_sentiment'] == 'very_negative'].copy()

index = 6993
data_augmentation_vertical_flip(negative, index, image_path)

index = 7473
data_augmentation_vertical_flip(very_negative, index, image_path)

index = 7624
data_augmentation_horizontal_flip(negative, index, image_path)

index = 8104
data_augmentation_horizontal_flip(very_negative, index, image_path)

index = 8255
data_augmentation_rotate(negative, index, image_path)

# File names sorting
def filenames(folder):
    images = []
    filenames = []
    
    for filename in os.listdir(folder):
        filenames.append(os.path.join(folder, filename))
    
    filenames = natsorted(filenames)
    return filenames

image_filenames = filenames(image_path)
print(len(image_filenames))

# Remove large images
large_images = []

for i in image_filenames:
    image = io.imread(i, as_gray=True)
    h, w = image.shape
    
    if h > 2000 or w > 2000:
        large_images.append(i)

new_array = []

for i in large_images:
    x = i.split('images/')
    new_array.append(x[1])

for i in new_array:
    df.drop(df.index[df['image_name'] == i], inplace=True)
    os.remove(os.path.join(image_path, i))

# Save cleaned labels
df.to_csv('clean.csv')

# Filter and extract features
def load_images(filenames):
    images = []
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    for i in filenames:
        new_path = os.path.join(image_path, i)
        image = io.imread(new_path, as_gray=True)
        resized_image = transform.resize(image, (500, 500))
        
        can = feature.canny(resized_image)
        can = can.flatten()
        images.append(can)
    
    return images

image_name = df['image_name']
images = load_images(image_name)

import pickle

rm = pickle.load(open('MoreFeatures.pkl', 'rb'))
pickle.dump(images, open('More_Features.pkl', 'wb'))

image = [torch.Tensor(i) for i in rm]

# Text feature extraction
texts = df['text_corrected'].tolist()
vectorizer = HashingVectorizer(n_features=50)
encoded = []

vect = vectorizer.fit_transform(df['text_corrected'])

for i in vect:
    encoded.append(i.toarray())

text = [i.flatten() for i in encoded]

# Combine image and text features
features = pd.DataFrame({'Image': images, 'Text': text})

# Map sentiment labels to numerical values using a dictionary
label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2, 'very_positive': 2, 'very_negative': 0}
df['overall_sentiment'] = df['overall_sentiment'].map(label_mapping)
labels = df['overall_sentiment']

# Split data into train and test sets
from sklearn.model_selection import train_test_split
ImageX_train, ImageX_test, ImageY_train, ImageY_test = train_test_split(features['Image'], labels, test_size=0.33, random_state=42)

TextX_train, TextX_test, TextY_train, TextY_test = train_test_split(features['Text'], labels, test_size=0.33, random_state=42)

# Convert data to PyTorch tensors
ImageX_trainTensor = torch.stack([torch.Tensor(x) for x in ImageX_train])
Imagey_trainTensor = torch.Tensor(ImageY_train.values)
ImageX_testTensor = torch.stack([torch.Tensor(x) for x in ImageX_test])
Imagey_testTensor = torch.Tensor(ImageY_test.values)
TextX_trainTensor = torch.stack([torch.Tensor(x) for x in TextX_train])
TextX_testTensor = torch.stack([torch.Tensor(x) for x in TextX_test])
TextY_trainTensor = torch.Tensor(TextY_train.values)
TextY_testTensor = torch.Tensor(TextY_test.values)

# Define the neural network architecture
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
        
        # Final Output Layer
        self.outputTogether = nn.Linear(250, 3)
    
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
        
        # Final output layer
        output = self.outputTogether(concatenate)
        
        return output

neural = ANN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(neural.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    prediction = []
    Labels = []
    loss_ = 0.0
    
    for i in range(len(ImageX_trainTensor)):
        optimizer.zero_grad()
        y_pred = neural(ImageX_trainTensor[i].cuda(), TextX_trainTensor[i].cuda()).unsqueeze(0)
        
        loss = criterion(y_pred, Imagey_trainTensor[i].unsqueeze(0).long().cuda()).unsqueeze(0)
        loss.backward()
        optimizer.step()
        loss_ += loss.item()
        
        if i % 2000 == 1999:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {loss_ / 2000:.3f}')
            loss_ = 0.0
    
    prediction = []
    Labels = []
    
    for i in range(len(ImageX_testTensor)):
        y_pred = neural(ImageX_testTensor[i].cuda(), TextX_testTensor[i].cuda()).unsqueeze(0)
        prediction.append(int(torch.argmax(y_pred)))
        Labels.append(int(Imagey_testTensor[i].long()))
    
    print(f'Epoch {epoch + 1}, Accuracy for Testing: {accuracy_score(Labels, prediction) * 100:.2f}%')
    print(f'Epoch {epoch + 1}, Report for Testing:\n{classification_report(Labels, prediction)}')
    print(f'Epoch {epoch + 1}, F1 Score for Testing: {f1_score(Labels, prediction, average="macro") * 100:.2f}%')

# Save the trained model
torch.save(neural.state_dict(), 'model.pth')
