import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()# Visualise inside a notebook
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

# Our spaCy model:
nlp = en_core_web_md.load()
# Tags I want to remove from the text
removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
tokens = []
for summary in nlp.pipe(reports['microtext1']):
   proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
   tokens.append(proj_tok)

   reports['tokens'] = tokens 
reports['tokens']

dictionary = Dictionary(reports['tokens']) #maps each token to a unique ID 
return (dictionary.token2id)

#limits low-frequency and high-frequency tokens

#No_below: tokens that appear in less than 5 documents are filtered out.
#No_above: tokens that appear in more than 50% of the total corpus are also removed as default.
#Keep_n: if n = 100, top 1000 most frequent tokens (default is 100.000). Set to ‘None’ if you want to keep all.

#dictionary.filter_extremes(no_below=2, no_above=0.7, keep_n= None) 

#counts the number of occurances of each distinct word

#converts the word to its integer word id and returns the result as a sparse vector

corpus = [dictionary.doc2bow(doc) for doc in reports['tokens']]
lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2, workers = 4, passes=5)

#computing cohrence c_v

topics = []
score = []
for i in range(1,20,1):
   lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=10, num_topics=i, workers = 4, passes=5, random_state=100)
   cm = CoherenceModel(model=lda_model, texts = reports['tokens'], corpus=corpus, dictionary=dictionary, coherence='c_v')
   topics.append(i)
   score.append(cm.get_coherence())
_=plt.plot(topics, score)
_=plt.xlabel('Number of Topics')
_=plt.ylabel('Coherence Score')
plt.show()

lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=100, num_topics=5, workers = 4, passes=100)

lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(lda_display)
