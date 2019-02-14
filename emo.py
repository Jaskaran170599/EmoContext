#Please use python 3.5 or above
import numpy as np
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from nltk import wordnet
from keras.layers import Dense, Embedding, LSTM
from keras import optimizers
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json, argparse, os
import re
from nltk.corpus import stopwords
import io
import nltk
from nltk import pos_tag
import sys
import string


# In[207]:


label2emotion = {0:"others", 1:"happy", 2: "sad", 3:"angry"}
emotion2label = {"others":0, "happy":1, "sad":2, "angry":3}
max_len=25

stw=stopwords.words("english")+['0','1','2','3','4','5','6','7','8','9']
stw_imp=['nor','not',"don't","aren't","hadn't","doesn't","wouldn't","won't","weren't","wasn't","shouldn't","shan't","needn't","mustn't"
,"mightn't","isn't","haven't","hasn't","didn't","couldn't"]
# In[208]:

final=[]
for word in stw:
    if word in stw_imp:
        continue
    final.append(word)
stop_words=final

emoji={}
emoji["-_-"]=":|"
emoji["😑"]=":|"
emoji["😭"]=':('
emoji["😀"]=":)"
emoji["😂"]='lol'
emoji["😍"]="loved"
emoji["😁"]=":)"


# In[209]:


def loadEmbeddings(embeddingfile,emb_dim):
    Embeddings={}
    fe = open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        Embeddings[word]=vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    Embeddings["<eos>"] = ("0.0 "*emb_dim).strip()
    Embeddings["<na>"]=("-1.0 "*emb_dim).strip()
    fe.close()
    return Embeddings


# In[210]:
wl=WordNetLemmatizer()


def get_tag(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.wordnet.ADV
    else:
        return wordnet.wordnet.NOUN
    
def get_lem(text):
    t=""
    
    text=pos_tag(text.strip().split())
#     print(text)
    for m in text:
        t+=wl.lemmatize(m[0].lower(),get_tag(m[1]))+" "
    return t.strip()


dictionary={}
def count_words(dataFilePath):
     with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            convo = ' <eos> '.join(line[1:4])
            convo+=' <eos>'
#             print(convo)
            conv=[]
            for word in convo.split(" "):
                if word in stop_words:
                    continue
                else:
                    conv.append(word)
            conv=get_lem(" ".join(conv))
                       
            # Remove any duplicate spaces
            conv=conv.replace("'","")
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            conv=conv.strip()
            for i in conv.split():
                if i not in dictionary.keys():
                        dictionary[i]=0
                else:
                    dictionary[i]+=1
        return dictionary
words_dict=count_words("./data/train.txt")

word_list=[(words_dict[i],i) for i in words_dict.keys()]
word_list=sorted(word_list,reverse=True)
dict_final=[]
for i,j in word_list[:3000]:
    dict_final.append(j)
for i in emoji.keys():
    dict_final.append(emoji[i])

def preprocessData(dataFilePath, mode,glove, sswe,batch_size,k=1):
    """Load data from a file, process and return indices, conversations and labels in separate lists
    Input:
        dataFilePath : Path to train/test file to be processed
        mode : "train" mode returns labels. "test" mode doesn't return labels.
    Output:
        indices : Unique conversation ID list
        conversations : List of 3 turn conversations, processed and each turn separated by the <eos> tag
        labels : [Only available in "train" mode] List of labels
    """
    num_of_zeros=batch_size//k
    indices = []
    conversations = []
    labels = []
    m=0
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            # Convert multiple instances of . ? ! , to single instance
            # okay...sure -> okay . sure
            # okay???sure -> okay ? sure
            # Add whitespace around such punctuation
            # okay!sure -> okay ! sure
            
            repeatedChars = ['.', '?', '!', ',']
            for c in repeatedChars:
                lineSplit = line.split(c)
                while True:
                    try:
                        lineSplit.remove('')
                    except:
                        break
                cSpace = ' ' + c + ' '    
                line = cSpace.join(lineSplit)
            
            line = line.strip().split('\t')
            if mode == "train":
                # Train data contains id, 3 turns and label
                label = emotion2label[line[4]]
                if (label==0 and num_of_zeros<=0):
                    continue
                labels.append(label)
                if label==0:
                    num_of_zeros-=1
            m+=1
            
            convo = ' <eos> '.join(line[1:4])
            convo+=' <eos>'
#             print(convo)
            conv=[]
            for word in convo.split(" "):
                if word in dict_final:
                    conv.append(word)
            conv=get_lem(" ".join(conv))
            
            # Remove any duplicate spaces
            conv=conv.replace("'","")
            duplicateSpacePattern = re.compile(r'\ +')
            conv = re.sub(duplicateSpacePattern, ' ', conv)
            
            conv=conv.strip()
            conv=" ".join(conv.split(" ")[:max_len])
            
            conv=(max_len-len(conv.split())) * '<eos> '+conv
            emb_conv_glove=[]
            emb_conv_sswe=[]
            
            for word in conv.strip().split(" "):
                if word in emoji:
                    word=emoji[word]
                if (word not in dict_final or word not in glove.keys()):
                    word="<na>"
                emb_conv_glove.append(np.array([float(i) for i in glove[word.lower()].split(" ")]))
                emb_conv_sswe.append(np.array([float(i) for i in sswe[word.lower()].split(" ")])) 
            
            indices.append(int(line[0]))
            conversations.append((np.array(emb_conv_glove),np.array(emb_conv_sswe)))
            if m==batch_size and mode=="train":
                num_of_zeros=batch_size//k
                yield(indices, np.array(conversations), np.array(labels))
                m=0
                
                del indices
                del conversations
                del labels
                indices=[]
                conversations=[]
                labels=[]
            elif m==batch_size:
                num_of_zeros=batch_size//k
                yield(indices, np.array(conversations))
                m=0
                del indices
                del conversations
                indices=[]
                conversations=[]
    num_of_zeros=batch_size//k
    if mode == "train":
        yield indices, np.array(conversations), labels
    else:
        yield indices, np.array(conversations)
# In[211]:


# Glove_emb=loadEmbeddings("./glove.6B.50d.txt",50)
# Sswe_emb=Glove_emb
# len(Glove_emb)


# In[212]:


# def fn(fn1):
#     for batch in fn1:
#         print(batch[1].shape)
#         break
# fn(preprocessData("./data/train.txt","train",Glove_emb,Glove_emb,256))


# In[213]:


def getMetrics(predictions, ground,NUM_CLASSES):
    """Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = to_categorical(predictions.argmax(axis=1))
    
    truePositives = np.sum(discretePredictions*ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground-discretePredictions, 0, 1), axis=0)
    
    print("True Positives per class : ", truePositives)
    print("False Positives per class : ", falsePositives)
    print("False Negatives per class : ", falseNegatives)
    
    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = ( 2 * recall * precision ) / (precision + recall) if (precision+recall) > 0 else 0
        print("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))
    
    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision ) / (macroPrecision + macroRecall) if (macroPrecision+macroRecall) > 0 else 0
    print("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))   
    
    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()    
    
    print("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))
    
    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)
    
    microF1 = ( 2 * microRecall * microPrecision ) / (microPrecision + microRecall) if (microPrecision+microRecall) > 0 else 0
    # -----------------------------------------------------
    
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions==ground)
    
    print("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (accuracy, microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


