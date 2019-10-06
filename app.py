import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import GaussianNB

traind = pd.read_csv('red.csv')


count= traind['condition']==0
check=traind[count]
check.count()

traind[traind.isnull().any(axis=1)].count()

traind = traind.sample(518, replace=False)

traind.head(10)

traind.shape

dbpedia_df = traind

X = dbpedia_df['sentence']
Y = dbpedia_df['condition']

count_vectorizer = CountVectorizer(min_df=0, max_df=80, ngram_range=(2, 2))

feature_vector = count_vectorizer.fit_transform(X)

feature_vector.shape

tfidf_transformer = TfidfTransformer()

feature_vector = tfidf_transformer.fit_transform(feature_vector)

feature_vector.shape

X_dense = feature_vector.todense()
X_dense.shape
x_train, x_test, y_train, y_test = train_test_split(X_dense, Y, test_size = 0.2)


import torch
import numpy as np

Xtrain_ = torch.from_numpy(x_train).float()
Xtest_ = torch.from_numpy(x_test).float()

Ytrain_ = torch.from_numpy(y_train.values).view(1,-1)[0]
Ytest_ = torch.from_numpy(y_test.values).view(1,-1)[0]  


import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 4939
output_size = 3
hidden_size = 10

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.fc3 = nn.Linear(hidden_size, output_size) 

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) 
        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)

model = Net()

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

loss_fn = nn.NLLLoss()

epoch_data = []
epochs = 1001

for epoch in range(1, epochs):

    optimizer.zero_grad()
    Ypred = model(Xtrain_)

    loss = loss_fn(Ypred , Ytrain_)
    loss.backward()

    optimizer.step()

    Ypred_test = model(Xtest_)
    loss_test = loss_fn(Ypred_test, Ytest_)

    _,pred = Ypred_test.data.max(1)

    accuracy = pred.eq(Ytest_.data).sum().item() / y_test.values.size
    epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), accuracy])


import pandas as pd
df_epochs_data = pd.DataFrame(epoch_data, 
                              columns=["epoch", "train_loss", "test_loss", "accuracy"])




app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    check = request.form.get('mindset')
    check = [check]
    check = count_vectorizer.transform(check).toarray()     

    check.shape 
    check.dtype
    sample = np.array(check,dtype='int64')
    sample.dtype
    sample_tensor = torch.from_numpy(sample).float()
    out = model(sample_tensor)
    out
    _, predicted = torch.max(out.data, -1)


    if predicted.item() == 0: 
        print("null -", predicted.item())
        output = 0
    elif predicted.item() == 1:
        print("inconclusive - ", predicted.item())
        output = 1
    elif predicted.item() == 2:
        print("conclusive - ", predicted.item())        
        output = 2

    return render_template('index.html', prediction_text= output)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
