#!/usr/bin/env python
# coding: utf-8

# In[ ]:



#get_ipython().system('pip install twilio')
#pip install twilio
from twilio.rest import Client

account_sid = 'AC225bee69b87720fa00cc282e477d8998'
auth_token = 'a406d3d14fbe6e7414103a98490667b3'
client = Client(account_sid, auth_token)

import nltk
nltk.download('punkt')
nltk.download('wordnet')


import numpy as np
import requests
from tensorflow.keras.models import Sequential
#from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.sentiment import SentimentIntensityAnalyzer
import json
import pickle
import flask

words=[]
classes = []
documents = []
ignore_letters = ['!', '?', ',', '.']
intents_file = open("intents.json").read()
intents = json.loads(intents_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
#print(documents)
# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
#print (len(documents), "documents")
# classes = intents
#print (len(classes), "classes", classes)
# words = all words, vocabulary
#print (len(words), "unique lemmatized words", words)

pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
        
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
# Create the model
model = Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

# Compile the model
optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Fit the model
history = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)

# Save the model
model.save('WBot')


from keras.models import load_model
model = load_model('WBot')
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl",'rb'))
classes = pickle.load(open("classes.pkl",'rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                #if show_details:
                    #print ("found in bag: %s" % word)
    return(np.array(bag))
def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents: 
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def responsed(msg1):
   # msg.append(msg1)
    ints = predict_class(msg1)
    response = getResponse(ints, intents)
    return response


def song_emotion(msgs):
    sid = SentimentIntensityAnalyzer()
    songs = {}

    for msg in msgs:
        sentiment_scores = sid.polarity_scores(msg)
        emotion = max(sentiment_scores, key=sentiment_scores.get)
        if(emotion == "neg"):
            emotion = "sad"
           # print(emotion)

        url = f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key=715156ee90d656693aaf300097923cbd&format=json&limit=20"
        response = requests.get(url)
        payload = response.json()

        for i in range(10):
            r = payload['tracks']['track'][i]
            songs[r['name']] = r['url']

    song_urls = list(songs.values())

    if len(song_urls) > 5:
        random.shuffle(song_urls)
        song_urls = song_urls[:5]

    return '\n'.join(song_urls)
    
from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)



@app.route('/sms', methods=['POST'])
def sms():
    msgs=[]
    flag = True
    while(flag == True):
        message_body = request.form['Body']
        message = message_body.lower()
        msgs.append(message)
        if len(msgs) > 15:
            msgs = msgs[-15:]
        if(message == "bye" ):
            flag = False
            response = responsed(message)
            send_message(response)
        else:
            response = responsed(message)
            if(response =="Here are some songs for you" or response == "right here for you" ):
                send_message(response)
                response = self.song_emotion(msgs)
            send_message(response)

def send_message(message):
    client.messages.create(
        body=message
    )

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    app.run() 






