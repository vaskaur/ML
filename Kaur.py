# -*- coding: utf-8 -*-
"""Untitled12.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GUCvaQVtwD4BC_zsPFrny-__AfIdCCu9
"""

import re
import nltk
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from google.colab import files

file = open("Tweets_train.txt","r")  
myfile = file.read()
sentences = myfile.split("\"\n")
X_train = []
Y_train = []
for sentence in sentences:
    Y_train.append(sentence.split("\t")[0])
    X_train.append(sentence.split("\t")[1])

X_train_nourls = []
for x in X_train:
     filterurl = re.sub(r"http\S+", "", x)
     filteruname = re.sub(r'@\w+', '', filterurl)
     filterhashtags = re.sub(r'#\w+', '', filteruname)
     #filterothers = re.sub(r'[^A-Za-z0-9]+', '', filterhashtags)
     X_train_nourls.append(filterhashtags)

X_train = X_train_nourls
#for s in X_train:
#print(s)

words = []
for x in X_train:
  words.append(x.split(" "))

#print(words)

filtered_sentences = [] 
  
for w in words:
  filtered_sentence = []
  for w1 in w: 
    if w1.lower() not in stop_words: 
      filtered_sentence.append(w1.lower())
  filtered_sentences.append(filtered_sentence) 
#print(filtered_sentences)


#print(words) 
#print(filtered_sentence) 

pos_dict = {}
#wlist = []*
for i in range(0,len(Y_train)):
  if Y_train[i] == '1':
      wlist = filtered_sentences[i]
      for i in wlist: 
        if i not in pos_dict.keys():
          pos_dict[i] = 0
        pos_dict[i] += 1
#del pos_dict['"']        
#print(pos_dict)

pos_words = sum(pos_dict.values())
#print(pos_words)

neg_dict= {}
for i in range(0,len(Y_train)):
  if Y_train[i] == '0':
      wlist = filtered_sentences[i]
      for i in wlist: 
        if i not in neg_dict.keys():
          neg_dict[i] = 0
        neg_dict[i] += 1
#del neg_dict['"']         
#print(neg_dict)

neg_words = sum(neg_dict.values())
#print(neg_words)
t_words = pos_words + neg_words

prob_p = pos_words/t_words
prob_n = neg_words/t_words

prob_dict_pos = {}
prob_dict_neg = {}

p_pd_key = []
p_pd_key = list(pos_dict.keys())
prob_v_pos = {}
for i in p_pd_key:
    prob_v_pos[i] = pos_dict.get(i)/pos_words

#print(prob_v_pos)


n_pd_key = []
n_pd_key = list(neg_dict.keys())
prob_v_neg = {}
for i in n_pd_key:
    prob_v_neg[i] = neg_dict.get(i)/neg_words

#print(prob_v_neg)

pos_sent_prob = []

for i in filtered_sentences:
  prob_word_p = 1.1
  for j in i:
    #for k in p_pd_key:
    if j in prob_v_pos.keys():
      prob_word_p = prob_p*prob_v_pos[j]*prob_word_p  
  pos_sent_prob.append(prob_word_p)

#print(pos_sent_prob)
#print(len(pos_sent_prob))
#print(len(Y_train))    

neg_sent_prob = []

for i in filtered_sentences:
  prob_word_n = 1.1
  for j in i:
    if j in prob_v_neg.keys():
      prob_word_n = prob_n*prob_v_neg[j]*prob_word_n  
  neg_sent_prob.append(prob_word_n)
#print(neg_sent_prob)
#print(len(neg_sent_prob))
#print(len(Y_train))   

predictions = []
for i in range(0,len(Y_train)):

  if pos_sent_prob[i] <= neg_sent_prob[i]:
      predictions.append('1')
  else:
      predictions.append('0')

mcount = 0
for i in range(0,len(Y_train)):
  if Y_train[i] == predictions[i]:
    mcount = mcount + 1
print((mcount/len(Y_train))*100)

'''
#Plot Accuracy for Training Data. Note: These accuracies are entred from the screeanshots generated for the data.
fig = plt.figure()
barg = fig.add_axes([0,0,1,1])
Edits = ['Raw', 'No URL', 'NoURL,Name', 'NoURL,Name,HTag', 'Cleaned']
Accuracy = [93, 92, 83, 82, 97]
barg.bar(Edits, Accuracy)
barg.set_ylabel('Accuracy')
barg.set_xlabel('Edits done on data')
plt.show()
'''

#####----Test----#####  
tfile = open("Tweets_test.txt","r")
myfilet = tfile.read()
sentencest = myfilet.split("\"\n")
X_traint = []
Y_traint = []
for sentencet in sentencest:
    Y_traint.append(sentencet.split("\t")[0])
    X_traint.append(sentencet.split("\t")[1])

X_train_nourlst = []
for x in X_traint:
     filterurlt = re.sub(r"http\S+", "", x)
     #filterunamet = re.sub(r'@\w+', '', filterurlt)
     #filterhashtagst = re.sub(r'#\w+', '', filterunamet)

     X_train_nourlst.append(filterurlt)

X_traint = X_train_nourlst

wordst = []
for x in X_traint:
  wordst.append(x.split(" "))    

filtered_sentencest = [] 
  
for w in wordst:
  filtered_sentencet = []
  for w1 in w: 
    if w1.lower() not in stop_words: 
      filtered_sentencet.append(w1.lower())
  filtered_sentencest.append(filtered_sentence) 


pos_sent_probt = []

for i in filtered_sentencest:
  prob_word_p = 1.1
  for j in i:
    #for k in p_pd_key:
    if j in prob_v_pos.keys():
      prob_word_p = prob_p*prob_v_pos[j]*prob_word_p  
  pos_sent_probt.append(prob_word_p)

#print(pos_sent_prob)
#print(len(pos_sent_prob))
#print(len(Y_train))    

neg_sent_probt = []

for i in filtered_sentencest:
  prob_word_n = 1.1
  for j in i:
    #for k in p_pd_key:
    if j in prob_v_neg.keys():
      prob_word_n = prob_n*prob_v_neg[j]*prob_word_n  
  neg_sent_probt.append(prob_word_n)
#print(neg_sent_prob)
#print(len(neg_sent_prob))
#print(len(Y_train))   

predictionst = []
for i in range(0,len(Y_traint)):

  if pos_sent_probt[i] >= neg_sent_probt[i]:
      predictionst.append('1')
  else:
      predictionst.append('0')

mcountt = 0
for i in range(0,len(Y_traint)):
  if Y_traint[i] == predictionst[i]:
    mcountt = mcountt + 1
acc=(mcountt/len(Y_traint))*100
print((mcountt/len(Y_traint))*100)

'''
#Plot Accuracy for Test data
plt.bar('Test Data', acc)
#plt.xticks(rotation = 90)
plt.title("Accuracy for Test Data")
plt.ylabel("Accuracy")
#plt.xlabel("Test Data")
'''

#------Election Tweets-------- 

filee = open("Tweets_Election_2.txt","r", encoding= 'utf-8')
myfilee = filee.read()
sentencese = myfilee.split("\n")

Y_trainfname = []
Y_trainname = [] 
Y_traindate = []
Y_traintf = []
Y_trainstate = []
X_traintweet = []

for sentencee in sentencese:
  if(len(sentencee.split("\t")) > 5):
    #Y_trainfname.append(sentencee.split("\t")[0])
    #Y_trainname.append(sentencee.split("\t")[1])
    sen_date = sentencee.split("\t")[2]
    Y_traindate.append(sen_date.split(" ")[0])
    Y_traintf.append(sentencee.split("\t")[3])
    Y_trainstate.append(sentencee.split("\t")[4])
    X_traintweet.append(sentencee.split("\t")[5])

#Y_traindateonly.append(Y_traindate.split(" ")) 

X_train_nourlse = []
for x in X_traintweet:
    filterurle = re.sub(r"http\S+", "", x)
    filterunamee = re.sub(r'@\w+', '', filterurle)

    X_train_nourlse.append(filterunamee)
X_traintweet = X_train_nourlse
#print(X_traintweet)


wordse = []
for x in X_traintweet:
  wordse.append(x.split(" "))
  
filtered_sentencese = [] 
  
for w in wordse:
  filtered_sentencee = []
  for w1 in w: 
    if w1.lower() not in stop_words: 
      filtered_sentencee.append(w1)
  filtered_sentencese.append(filtered_sentencee) 
  
#print(words) 
#print(filtered_sentencese) 

pos_sent_prob_E = []

for i in filtered_sentencese:
  prob_word_p = 1.1
  for j in i:
    #for k in p_pd_key:
    if j in prob_v_pos.keys():
      prob_word_p = prob_p*prob_v_pos[j]*prob_word_p  
  pos_sent_prob_E.append(prob_word_p)

#print(pos_sent_prob)
#print(len(pos_sent_prob))
#print(len(Y_train))    

neg_sent_prob_E = []

for i in filtered_sentencese:
  prob_word_n = 1.1
  for j in i:
    #for k in p_pd_key:
    if j in prob_v_neg.keys():
      prob_word_n = prob_n*prob_v_neg[j]*prob_word_n  
  neg_sent_prob_E.append(prob_word_n)
#print(neg_sent_prob)
#print(len(neg_sent_prob))
#print(len(Y_train))   

predictions_E = []
for i in range(0,len(Y_traindate)):

  if pos_sent_prob_E[i] >= neg_sent_prob_E[i]:
      predictions_E.append('1')
  else:
      predictions_E.append('0')
#print(predictions_E)
#print(len(predictions_E))
#print(len(Y_traindate))

'''
#positvie and negative tweets vs frequency

count_pos_tweet = 0
count_neg_tweet = 0
print(len(predictions_E))
for i in range(0,len(predictions_E)):
 
  if predictions_E[i] == '1':
      count_pos_tweet += 1
  if predictions_E[i] == '0':
      count_neg_tweet += 1    

sentiment = ['Positive', 'Negative']
freq = [count_pos_tweet, count_neg_tweet]
plt.bar(sentiment, freq)
plt.xticks(rotation = 90)
plt.title("Tweets frequency according to sentiments")
plt.ylabel("Frequency")
plt.xlabel("Sentiment")

'''

'''
#pie chart for pos neg tweets for candidates
pos_count_joe = 0
pos_count_trump = 0 
neg_count_joe = 0
neg_count_trump = 0

for i in range(0,len(predictions_E)):
   
  if predictions_E[i] == '1':
    for k in filtered_sentencese[i]:
      #for k in j:
        if k == "Joe" or k =="Biden":
          pos_count_joe += 1
        if k == "Donald" or k == "Trump":
          pos_count_trump += 1
  if predictions_E[i] == '0':
    for k in filtered_sentencese[i]:
      #for k in j:
        if k == "Joe" or k =="Biden":
          neg_count_joe += 1
        if k == "Donald" or k == "Trump":
          neg_count_trump += 1       
#print(pos_count_joe)
#print(pos_count_trump)          
#print(pos_count_joe+pos_count_trump)
#print(neg_count_joe+neg_count_trump)

#Pie chart positive tweets
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
participants = ['Joe Biden', 'Donald Trump']
pos_outcomes = [pos_count_joe, pos_count_trump]
number = [pos_count_joe, pos_count_trump]
colours = ['pink', 'c']
ax.pie(pos_outcomes, labels = number, shadow=True, colors = colours)
ax.set_title("Number of Positive Tweets")
ax.legend(participants)
plt.show()

#Pie chart negative tweets
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
participants = ['Joe Biden', 'Donald Trump']
pos_outcomes = [neg_count_joe, neg_count_trump]
number = [neg_count_joe, neg_count_trump]
colours = ['pink', 'c']
ax.pie(pos_outcomes, labels = number, shadow=True, colors = colours)
ax.set_title("Number of Negative Tweets")
ax.legend(participants)
plt.show()
'''
'''
#Retweets
pos_count_joer = 0
pos_count_trumpr = 0 
neg_count_joer = 0
neg_count_trumpr = 0

for i in range(0,len(predictions_E)):
   
  if predictions_E[i] == '1':
    if Y_traintf[i] == 'True':
      for k in filtered_sentencese[i]:
      #for k in j:
          if k == "Joe" or k =="Biden":
            pos_count_joer += 1
          if k == "Donald" or k == "Trump":
            pos_count_trumpr += 1
  if predictions_E[i] == '0':
    if Y_traintf[i] == 'True':
      for k in filtered_sentencese[i]:
      #for k in j:
          if k == "Joe" or k =="Biden":
            neg_count_joer += 1
          if k == "Donald" or k == "Trump":
            neg_count_trumpr += 1   



labels = ['Positive re-tweet', 'Negative re-tweet']
trumprt = [pos_count_trumpr, neg_count_trumpr]
joert = [pos_count_joer, neg_count_joer]

x = np.arange(len(labels))  
width = 0.3

fig, ax = plt.subplots()
redj = ax.bar(x - width/2, trumprt, width, label='Donald Trump', color = 'c')
rejb = ax.bar(x + width/2, joert, width, label='Joe Biden', color = 'pink')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Re-tweets')
ax.set_title('Sentimenal re-tweets for candidates')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar(redj, padding=3)
ax.bar(rejb, padding=3)

fig.tight_layout()

plt.show()
'''


'''
# plot number of negative/positive tweets in a day
pos_count = []
neg_count = []
for i in range(0,len(predictions_E)):  
  if predictions_E[i] == '1':
        pos_count.append(Y_traindate[i])
  else:neg_count.append(Y_traindate[i]) 
counter_pos = {}
counter_neg = {}  
counter_pos = Counter(pos_count)
counter_neg = Counter(neg_count)

#print(counter_pos)
print(counter_neg)  

plt.bar(counter_neg.keys(), counter_neg.values(), color = 'pink')
plt.xticks(rotation = 90)
plt.title("Negative Tweets in a day")
plt.ylabel("Number of negative tweets")
plt.xlabel("Dates")


#positive and neg tweet in a day according to cadidate name
pos_count_Joe_date = []
pos_count_DT_date = []
neg_count_Joe_date = []
neg_count_DT_date = []
for i in range(0,len(predictions_E)):
  if predictions_E[i] == '1':
    for k in filtered_sentencese[i]:
        if k == "Joe" or k =="Biden":  
           pos_count_Joe_date.append(Y_traindate[i])
        if k == "Donald" or k =="Trump":
           pos_count_DT_date.append(Y_traindate[i])   
  if predictions_E[i] == '0':
    for k in filtered_sentencese[i]:
        if k == "Joe" or k =="Biden":  
           neg_count_Joe_date.append(Y_traindate[i])
        if k == "Donald" or k =="Trump":   
           neg_count_DT_date.append(Y_traindate[i])

counter_pos_Jdate = {}
counter_pos_Ddate = {}  
counter_pos_Jdate = Counter(pos_count_Joe_date)
counter_pos_Ddate = Counter(pos_count_DT_date)

counter_neg_Jdate = {}
counter_neg_Ddate = {}  
counter_neg_Jdate = Counter(neg_count_Joe_date)
counter_neg_Ddate = Counter(neg_count_DT_date)

#print(sum(counter_neg_Jdate.values()))
#print(sum(counter_neg_Ddate.values()))  


plt.bar(counter_neg_Jdate.keys(), counter_neg_Jdate.values(), color = 'pink')
plt.xticks(rotation = 90)
plt.title("Negative Tweets in a day for Joe Biden")
plt.ylabel("Number of tweets")
plt.xlabel("Dates")

plt.bar(counter_neg_Ddate.keys(), counter_neg_Ddate.values(), color = 'c')
plt.xticks(rotation = 90)
plt.title("Negative Tweets in a day for Donald Trump")
plt.ylabel("Number of tweets")
plt.xlabel("Dates")

plt.bar(counter_pos_Ddate.keys(), counter_pos_Ddate.values(), color = 'c')
plt.xticks(rotation = 90)
plt.title("Positive Tweets in a day for Donald Trump")
plt.ylabel("Number of tweets")
plt.xlabel("Dates")

plt.bar(counter_pos_Jdate.keys(), counter_pos_Jdate.values(), color = 'pink')
plt.xticks(rotation = 90)
plt.title("Positive Tweets in a day for Joe Biden")
plt.ylabel("Number of tweets")
plt.xlabel("Dates")

#Plots for negative and positive tweets according to states
pos_count_Joe_state = []
pos_count_DT_state = []
neg_count_Joe_state = []
neg_count_DT_state = []
for i in range(0,len(predictions_E)):  
  if predictions_E[i] == '1':
     for k in filtered_sentencese[i]:
         if k == "Joe" or k =="Biden":  
           pos_count_Joe_state.append(Y_trainstate[i])
         if k == "Donald" or k =="Trump":
           pos_count_DT_state.append(Y_trainstate[i])   
  if predictions_E[i] == '0':
    for k in filtered_sentencese[i]:
        if k == "Joe" or k =="Biden":  
           neg_count_Joe_state.append(Y_trainstate[i])
        if k == "Donald" or k =="Trump":   
           neg_count_DT_state.append(Y_trainstate[i])
counter_pos_Jstate = {}
counter_pos_Dstate = {}  
counter_pos_Jstate = Counter(pos_count_Joe_state)
counter_pos_Dstate = Counter(pos_count_DT_state)

counter_neg_Jstate = {}
counter_neg_Dstate = {}  
counter_neg_Jstate = Counter(neg_count_Joe_state)
counter_neg_Dstate = Counter(neg_count_DT_state)

fig = plt.figure(figsize=[70,7], dpi=80)
plt.bar(counter_pos_Dstate.keys(), counter_pos_Dstate.values(), color ='c')
plt.xticks(rotation = 90)
plt.title("Positive Tweets from various states for Donanld Trump")
plt.ylabel("Number of tweets")
plt.xlabel("States")
#plt.savefig("election_neg_dt.png")
#files.download("election_neg_dt.png")

fig = plt.figure(figsize=[70,7], dpi=80)
plt.bar(counter_neg_Dstate.keys(), counter_neg_Dstate.values(), color ='c')
plt.xticks(rotation = 90)
plt.title("Negative Tweets from various states for Donanld Trump")
plt.ylabel("Number of negative tweets")
plt.xlabel("States")
#plt.savefig("election_neg_dt.png")
#files.download("election_neg_dt.png")

fig = plt.figure(figsize=[70,7], dpi=80)
plt.bar(counter_pos_Jstate.keys(), counter_pos_Jstate.values(), color ='pink')
plt.xticks(rotation = 90)
plt.title("Positive Tweets from various states for Joe Biden")
plt.ylabel("Number of tweets")
plt.xlabel("States")
#plt.savefig("election_neg_dt.png")
#files.download("election_neg_dt.png")

fig = plt.figure(figsize=[70,7], dpi=80)
plt.bar(counter_neg_Jstate.keys(), counter_neg_Jstate.values(), color ='pink')
plt.xticks(rotation = 90)
plt.title("Negative Tweets from various states for Joe Biden")
plt.ylabel("Number of tweets")
plt.xlabel("States")
#plt.savefig("election_neg_dt.png")
#files.download("election_neg_dt.png")
'''