from typing import List, Dict, Counter, Iterable, Tuple
import collections
import heapq
from tqdm import tqdm

import numpy as np
import numpy.typing as npt

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer


Text = str
LemmatizedText = str
Sentence = str
Word = str
WordRoot = str
TreeBankTag = str
WordNetTag = str
Token = int
Index = int
Label = int
ColumnVector = npt.NDArray
RowVector = npt.NDArray
Dictionary = Dict[Word, Token]


def treebankToWordnetPOS(treebankTag: TreeBankTag) -> WordNetTag:
    if treebankTag.startswith("J"):
        return wordnet.ADJ
    if treebankTag.startswith("V"):
        return wordnet.VERB
    if treebankTag.startswith("N"):
        return wordnet.NOUN
    if treebankTag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
def lemmatize(docs: Iterable[Text]) -> List[LemmatizedText]:
    
    lemmatizer = WordNetLemmatizer()
    
    output = []
    for rawDoc in tqdm(docs, desc = "lemmatizing", position=0, disable=True):
        # break, get tags, remake sentence
        wordsAndTags = nltk.pos_tag(word_tokenize(rawDoc))
        lemmatizedWs = [lemmatizer.lemmatize(word, treebankToWordnetPOS(tag)) 
                            for word, tag in wordsAndTags]
        output.append(" ".join(lemmatizedWs))
    
    return output
    
    
def getWordFrequency(
        tokenizer, 
        docs: Iterable[LemmatizedText],
        lowercase=True,
        ignorePunkt=True
    ) -> Counter[WordRoot]:
    
    fMap = collections.Counter()
    for doc in tqdm(docs, desc="counting words", position=0, disable=True):
        words = tokenizer(doc)
        if lowercase:
            words = [word.lower() for word in words]
        if ignorePunkt:
            words = [word for word in words if word.isalnum()]
        fMap.update(words)
    return fMap
    
def getTopWords(
        tokenizer, 
        docs: Iterable[LemmatizedText], 
        maxSize=2000, 
        stopWords=None,
        lowercase=True,
        ignorePunkt=True
    ) -> List[WordRoot]:
    # frequency of words
    fMap = getWordFrequency(tokenizer, docs, lowercase=lowercase, ignorePunkt=ignorePunkt)
    
    # remove stop words
    if stopWords is not None:
        for stopWord in tqdm(stopWords, desc="removing stop words", position=0, disable=True):
            del fMap[stopWord]
        
    # choose maxSize words based on frequency
    topWords =  heapq.nlargest(maxSize, fMap.keys(), fMap.__getitem__)
    print(f"*** top 10 words out of {len(fMap)} words***")
    for i in range(10):
        print(topWords[i], fMap[topWords[i]])
    return topWords

def buildWordToIndex(
        tokenizer, 
        docs: Iterable[LemmatizedText], 
        maxSize=2000, 
        stopWords=None,
        lowercase=True,
        ignorePunkt=True
    ) -> Dict[WordRoot, Index]:
    
    topWords = getTopWords(
        tokenizer, 
        docs, 
        maxSize = maxSize, 
        stopWords = stopWords,
        lowercase = lowercase,
        ignorePunkt = ignorePunkt
    )
    
    wToI = {}
    index = 0
    for w in topWords:
        wToI[w] = index
        index += 1
    wToI["UNK"] = len(wToI)
    return wToI

def lowerUnPunkt(words: List[Word]) -> List[Word]:
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalnum()]
    return words

def tokenizeDoc(tokenizer, doc: LemmatizedText, wordToIndex: Dict[WordRoot, Token]) -> List[Token]:
    words = tokenizer(doc)
    words = lowerUnPunkt(words)
    return [wordToIndex[w] 
                if w in wordToIndex 
                else wordToIndex["UNK"]
            for w in words]


# def getTermFreqDoc(doc: str, wordToIndex: Dict[str, int]):
#     fMap = collections.Counter(doc)
#     idxFreq = [0] * 
#     for w, wIndex in wordToIndex.items():
        

def getTermFreqMatrix(tokenizer, docs: Iterable[LemmatizedText], wordToIndex: Dict[WordRoot, Token], lowercase=True) -> np.ndarray:
    tf = np.zeros((len(docs), len(wordToIndex)))
    
    row = 0
    for doc in docs:
        fMap = getWordFrequency(tokenizer, [doc], lowercase=lowercase)
        # print(fMap)
        # assuming dictionary has more unique words than a doc, we iterate over dic words
        for dicWord, freq in fMap.items():
            if lowercase:
                dicWord = dicWord.lower()
            if dicWord in wordToIndex:
                wordIndex = wordToIndex[dicWord]
                # print(dicWord, freq, row, wordIndex)
                tf[row][wordIndex] = freq
                
        row += 1
    
    return tf
        
    
def getDf(tf: npt.NDArray) -> npt.NDArray:
    exists = tf > 0
    return exists.sum(axis=0)


def oneHotColumn(word: Word, wordToIndex: Dictionary) -> ColumnVector:
    hot = np.zeros((len(wordToIndex), 1))
    hotIdx = wordToIndex[word]
    hot[hotIdx] = 1
    return hot
    

