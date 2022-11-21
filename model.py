import json
import string
import random 
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")
nltk.download('omw-1.4')

data = {"intents": [
            {"tag": "name",
            "patterns": ["hvad er dit navn",
            "hvem er du",
            "navn tak"],
            "responses": ["Jeg er din AI assistent"]
            },
            {"tag": "greeting",
            "patterns": [ "hej",
            "er der nogen",
            "hvad så",
            "Hej",
            "jo",
            "Lyt",
            "Vær venlig at hjælpe mig",
            "Jeg lærer af",
            "Jeg hører til",
            "mål batch",
            "hej du",
            "taler med dig for første gang"],
            "responses": ["Hej! Hvordan kan jeg hjælpe dig?"],
            },
            {"tag": "hru",
            "patterns": ["godt tak",
                        "det går godt",
                        "tak jeg har det godt",
                        "det går super",
                        "har det fantastisk",
                        "tak går godt",
                        "Godt"],
            "responses": ["Det er jeg glad for at høre", "dejligt at høre", "fantastisk!"]
            },
            {"tag": "hru-back",
            "patterns": ["har du det godt",
                        "er det godt at være en AI bot"],
            "responses": ["Jeg har det super, det er fantastisk at være en Bot"]
            },
            {"tag": "goodbye",
            "patterns": [ "tak skal du have",
            "tak",
            "cya",
            "vi ses",
            "senere",
            "vi ses senere",
            "farvel",
            "Jeg forlader",
            "hav en god dag",
            "du hjalp mig",
            "mange tak",
            "tusind tak",
            "du er den bedste",
            "god hjælp",
            "for godt",
            "du er en god lærekammerat"],
            "responses": ["Jeg håber at jeg var dig behjælpelig. Farvel"]
            },
        
]}

def tokenize_text(txt):
    tokens = nltk.word_tokenize(txt)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def word_bag(txt, vocab):
    tokens = tokenize_text(txt)
    bag_words = [0] * len(vocab)
    for i in tokens:
        for idx, word in enumerate(vocab):
            if word == i:
                bag_words[idx] = 1
    return num.array(bag_words)

def predict_class(txt, vocab, labels):
    bag_words = word_bag(txt, vocab)
    result = model.predict(num.array([bag_words]))[0]
    threshold = 0.2
    y_prob = [[idx, res] for idx, res in enumerate(result) if res > threshold]

    y_prob.sort(key=lambda x: x[1], reverse=True)
    list_y = []
    for i in y_prob:
        list_y.append(labels[i[0]])
    return list_y

def get_result(result_list, json_intents):
    tag = result_list[0]
    list_intents = json_intents["intents"]
    for i in list_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

lemmatizer = WordNetLemmatizer()
data_classes = []
words = []
x = []
y = []
for i in data["intents"]:
  for j in i["patterns"]:
    tokens = nltk.word_tokenize(j)
    words.extend(tokens)
    x.append(j)
    y.append(i["tag"])
  
  if i["tag"] not in data_classes:
    data_classes.append(i["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words = sorted(set(words))
data_classes = sorted(set(data_classes))

training_data = []
output = [0] * len(data_classes)
for i,j in enumerate(x):
  bag_of_words = []
  text = lemmatizer.lemmatize(j.lower())
  for word in words:
    bag_of_words.append(1) if word in text else bag_of_words.append(0)
  
  output_row = list(output)
  output_row[data_classes.index(y[i])] = 1
  training_data.append([bag_of_words, output_row])

random.shuffle(training_data)
training_data = num.array(training_data, dtype=object)

xp = num.array(list(training_data[:, 0]))
yp = num.array(list(training_data[:, 1]))

shape_x = (len(xp[0]),)
shape_y = len(yp[0])

model = Sequential()
model.add(Dense(256, input_shape = shape_x, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.3))
model.add(Dense(shape_y, activation = "softmax"))

# Compile the model
model.compile(loss='categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001, decay = 1e-6), 
              metrics=['accuracy'])
print(model.summary())

# Train the model
model.fit(xp, yp, epochs=200, verbose=1)

print("")
print("Venter...")
# AI-chat bot input/output
while True:
    message = input("")
    intents = predict_class(message, words, data_classes)
    res = get_result(intents, data)
    print(res)