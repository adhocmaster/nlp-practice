I love to write beautiful code. So, I implemented every piece for code myself even when I followed some tutorial. **Let there be beautiful ML code.** However, the code is not written for performance and tries to use python data structures for readability. *The sole purpose of the codes is learning and not being used in production.*

# Getting Started:
Each notebook sets the current working directory to the root of the repository. set **projectFolder** in **init_notebook.py** to the root of the repository before running a notebook. The first cell of the notebooks needs to be run only once per session (you will get an error if the working directory is already changed once). This setting helps us to organize libraries into different folders and access them from anywhere inside the repository.

# 1. Building Vocabulary
- Stemming, Lemmatization, POS [Preprocessing](./preprocessing)
- Vocabulary, word frequence, TD-IDF [common_preprocess.py](./library/common_preprocess.py)

# 2. Tasks

## 2.1 Word Embedding
1. [Using Pretrained models](./neural/pretrained_embedding.ipynb)
2. Debiasing


## 2.2 Classical Methods:
1. [Document Classification with Bag-of-Words](./tasks/document-classfication-bow.ipynb)
2. [Movie recommender with TD-iDF](./tasks/movie-recommender-tfidf.ipynb)
3. [Text Classification with Naive Bayes](./tasks/text-classifier-bayes-poet.ipynb)
4. [Text Generation with Naive Bayes](./tasks/text-generator-bayes-poet.ipynb)

## 2.3 NN Methods

### Fine-tuning
1. [Fine Tuning Llama2 with LoRA](./tasks/fine-tuning/LoRA-Llama2.ipynb)