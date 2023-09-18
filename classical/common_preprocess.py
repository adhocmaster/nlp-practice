from typing import List, Dict, Counter, Iterable, Tuple
import collections
import heapq
from tqdm import tqdm

import numpy as np

import nltk
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer


def treebankToWordnetPOS(treebankTag: str) -> str:
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
    
    
def lemmatize(docs: Iterable[str]) -> List[str]:
    
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
        docs: Iterable[str],
        lowercase=True,
        ignorePunkt=True
    ) -> Counter[str]:
    
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
        docs: Iterable[str], 
        maxSize=2000, 
        stopWords=None,
        lowercase=True,
        ignorePunkt=True
    ) -> List[str]:
    # frequency of words
    fMap = getWordFrequency(tokenizer, docs)
    
    # remove stop words
    if stopWords is not None:
        for stopWord in tqdm(stopWords, desc="removing stop words", position=0, disable=True):
            del fMap[stopWord]
        
    # choose maxSize words based on frequency
    topWords =  heapq.nlargest(maxSize, fMap.keys(), fMap.__getitem__)
    for i in range(10):
        print(topWords[i], fMap[topWords[i]])
    return topWords

def buildWordToIndex(
        tokenizer, 
        docs: Iterable[str], 
        maxSize=2000, 
        stopWords=None,
        lowercase=True,
        ignorePunkt=True
    ) -> Dict[str, int]:
    
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
    
    return wToI

# def getTermFreqDoc(doc: str, wordToIndex: Dict[str, int]):
#     fMap = collections.Counter(doc)
#     idxFreq = [0] * 
#     for w, wIndex in wordToIndex.items():
        

def getTermFreqMatrix(tokenizer, docs: Iterable[str], wordToIndex: Dict[str, int], lowercase=True) -> np.ndarray:
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
    

