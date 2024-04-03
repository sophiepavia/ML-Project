# AdvML Final Project

## Libraries
```bash
import numpy as np
import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
import nltk
from nltk.corpus import reuters, wordnet as wn
from nltk.corpus import stopwords
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import gensim.downloader
import inflect
from gensim.models import Word2Vec
```

After installing the NLTK package, please do install the necessary datasets/models for specific functions to work. (reuters)

On the command line type python -m nltk.downloader reuters, or in the Python interpreter import nltk; nltk.download('reuters')

## Executing 
Simply run notebook for model of choice.
- LDA without Word Embeddings:

```bash
LDA-gibbs.ipynb
```

- LDA with Word Embeddings:

```bash
LDA-gibbs-wordEmbed.ipynb
```

## Source code
```
LDA.py
```

## Resources
https://naturale0.github.io/2021/02/16/LDA-4-Gibbs-Sampling

https://medium.com/@datastories/parameter-estimation-for-latent-dirichlet-allocation-explained-with-collapsed-gibbs-sampling-in-1d2ec78b64c
