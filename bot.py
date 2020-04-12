import nltk
import warnings
warnings.filterwarnings("ignore")
import random
import string

f=open('data.txt','r',errors = 'ignore')
m=open('info.txt','r',errors = 'ignore')
checkpoint = "./chatbot_weights.ckpt"

raw=f.read()
rawone=m.read()
raw=raw.lower()
rawone=rawone.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)
sent_tokensone = nltk.sent_tokenize(rawone)
word_tokensone = nltk.word_tokenize(rawone)


sent_tokens[:2]
sent_tokensone[:2]

word_tokens[:5]
word_tokensone[:5]

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

Introduce_Ans = ["My name is Janvi.", "My name is Janvi you can called me jan .", "I m Janvi :) ",
                 "My name is Janvi. and my nickname is Jan and i am happy to solve your queries :) "]

GREETING_INPUTS = ("hello", "hi", "hiii", "hii", "hiiii", "hiiii", "greetings", "sup", "what's up", "hey","hlo")
GREETING_RESPONSES = ["hi", "hey", "hii there", "hi there", "hello", "I am glad! You are talking to me"]

GOODBYE_INPUTS = ("cya", "See you later", "Goodbye", "I am Leaving", "Have a Good day", "bye")
GOODBYE_RESPONSES = ["Sad to see you go :", "Talk to you later", "Goodbye!"]

AGE_Q = ["what is your age?", "How old are you", "how old are you?", "what about your age?.","Your age?","your age?","What is your age?","your age?"]
AGE_Ans = ["I am few days old!", "Younger than you, so don't be jealous now. hahiahahi"]

EXAMS_Q = ("Do you have any information regarding exams?", "Exams information","Date sheet for exams ","do you any information regarding exams?","exams information","date sheet for exams","Exam information",
           "exam information")
EXAMS_Ans = ["Due to Lockdown for COVID19 all exams are postponded but be prepared for them.","All exams are postponded till further notification but be prepared for them"]


# Checking for greetings
def greeting(sentence):

    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Checking for Introduce
def IntroduceMe(sentence):
    return random.choice(Introduce_Ans)

# Checking for AGE_Q
def AGE(sentence):
    for word in AGE_Q:
        if sentence.lower() == word:
            return random.choice(AGE_Ans)


# Checking for Exams_Q
def EXAM(sentence):

    for word in EXAMS_Q:
        if sentence.lower() == word:
            return random.choice(EXAMS_Ans)


def goodbye(sentence):

    for word in sentence.split():
        if word.lower() in GOODBYE_INPUTS:
            return random.choice(GOODBYE_RESPONSES)



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response
      
# Generating response
def responseone(user_response):
    robo_response=''
    sent_tokensone.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokensone)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokensone[idx]
        return robo_response

def chat(user_response):
    user_response = user_response.lower()
    keyword = "age"
    keywordone = " age"
    keywordsecond = "age "

    if (user_response != 'bye'):
        if (user_response == 'thanks' or user_response == 'thank you'):
            flag = False

            return "You are welcome.."
        elif (AGE(user_response) != None):
            return AGE(user_response)
        else:
            if (user_response.find(keyword) != -1 or user_response.find(keywordone) != -1 or user_response.find(
                    keywordsecond) != -1):

                return responseone(user_response)
                sent_tokensone.remove(user_response)
            elif (greeting(user_response) != None):

                return greeting(user_response)

            elif(goodbye(user_response) != None):
                return goodbye(user_response)

            elif (user_response.find("your name") != -1 or user_response.find(" your name") != -1 or user_response.find(
                    "your name ") != -1 or user_response.find(" your name ") != -1):
                return IntroduceMe(user_response)

            elif (EXAM(user_response) != None):
                return EXAM(user_response)



                return response(user_response)
                sent_tokens.remove(user_response)

    else:
        flag = False

        return "Bye! take care.."
