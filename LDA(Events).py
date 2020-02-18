import nltk
import array
import itertools, collections
def consume(iterator, n):
    collections.deque(itertools.islice(iterator, n))

# c=['Wedding','engagement','RelationshipBegin','Anniversary','Devoice','Graduating','Admission','Exam','Research','Essay','Thesis','Job','Interview','Internship','Moving','Travel','Vacation','Winning Award']
# Stemming
import nltk
from nltk.stem.porter import *
stemmer = PorterStemmer()
from textblob import TextBlob




#////////////////////////// FOR POSITIVE DATASET//////////////////////////////////////////////////////////////
file4=open("final.txt","w")
# file read
f = open("pos.txt","r")
pos_data = f.read()

# extract real tweets and stemmed tweets
tweets = pos_data.split("@")
stemmed_tweets = []
for i in tweets:
    a = i.split()
    b = ""
    for j in a:
        b = b + " " + stemmer.stem(j)
    stemmed_tweets.append(b)
# Tokenize
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95,min_df=2,lowercase=True,stop_words='english')
dictionary = cv.fit_transform(stemmed_tweets)
dictionary


# LDA
from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=4, random_state=50)
LDA.fit(dictionary)

for i, topic in enumerate(LDA.components_):
    print(f"The top five words for topic #{i}")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-10:]])
    print('\n')
    print('\n')
arr=array.array('i', [])
topic_result = LDA.transform(dictionary)
for i, j in enumerate(tweets):
    file4.write('\n')
    file4.write(j)
    file4.write('\n')
    arr.append(topic_result[i].argmax())
    file4.write("Cluster No:")
    file4.write(str(topic_result[i].argmax()))
    file4.write('\n')




cf=0
df=0
ef=0
iterator = range(1, 787).__iter__()
for i in iterator:
  ef=ef+1
  if i==0:
      print("ok")
  elif arr[i]==arr[i+1]:
      print("Match")
      cf=cf+1
      consume(iterator,1)
  else:
      print("NOT MATCH")
      df=df+1
      consume(iterator, 1)


print("True Positive")
print('\n')
print(cf)
print('\n')
print("True Negative")
print('\n')
print(df)
print("itertions")
print(ef)

Acuuracy=0.0
Acuuracy=cf/ef*100
file4.write("------Accuracy----")
file4.write('\n')
file4.write(str(Acuuracy))

#





#////////////////////////////////// For Negative dataset//////////////////////////////////////////////////////////////////

f = open("neg.txt","r")
pos_data = f.read()
cf=0
df=0
ef=0
total=0
file5=open("nego.txt","w")
# extract real tweets and stemmed tweets
tweets = pos_data.split("RT")
stemmed_tweets = []
for i in tweets:
     i = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])| (\w+:\ / \ / \S+)", " ", i).split())
     i = re.sub(r"^http://t.co/[a-zA-Z0-9]*\s"," ",i)
     i = re.sub(r"\s+http://t.co/[a-zA-Z0-9]*\s", " ", i)
     i = re.sub(r"\s+http://t.co/[a-zA-Z0-9]*$"," ",i)
     i = i.lower()
     i = re.sub(r"that's","that is",i)
     i = re.sub(r"there's", "there is", i)
     i = re.sub(r"what's", "what is", i)
     i = re.sub(r"i'm","i am",i)
     i = re.sub(r"he's", "he is", i)
     i = re.sub(r"she's", "she is", i)
     i = re.sub(r"they're", "they are", i)
     i = re.sub(r"who're","who are",i)
     i = re.sub(r"ain't", "am not", i)
     i = re.sub(r"wouldn't", "would not", i)
     i = re.sub(r"shouldn't", "should not", i)
     i = re.sub(r"can't", "can not", i)
     i = re.sub(r"couldn't", "could not", i)
     i = re.sub(r"won't", "will not", i)
     i = re.sub(r"don't", "do not", i)
     i = re.sub(r"didn't", "did not", i)
     i = re.sub(r"\W", " ", i)
     i = re.sub(r"\d", " ", i)


     tweet = TextBlob(i)
     total=total+1
     file5.write('\n')
     file5.write(str(tweet))
     file5.write('\n')
     file5.write(str(tweet.sentiment))
     file5.write('\n')
     if tweet.sentiment[0] > 0:
         cf=cf+1
         file5.write('Positive')
         file5.write('\n')
     elif tweet.sentiment[0] < 0:
         file5.write('Negative')
         df=df+1
         file5.write('\n')
     else:
         file5.write('Neutral')
         ef=ef+1
         file5.write('\n')


positive=cf/total*100
negative=df/total*100
neutral=ef/total*100

print(positive)
print('\n')
print(negative)
print('\n')
print(neutral)

#
print(cf)
print('\n')
print(df)
print('\n')
print(ef)
print('\n')
print(total)
