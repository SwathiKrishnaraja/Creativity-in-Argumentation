import pandas as pd
from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = English()

with open("micro_b001.txt") as f:
    text=f.read()

#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)

# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)

from spacy.lang.en.stop_words import STOP_WORDS

# Create list of word tokens after removing stopwords
filtered_sentence =[] 

for word in token_list:
    lexeme = nlp.vocab[word]
    if lexeme.is_stop == False:
        filtered_sentence.append(word) 
print(token_list)
print(filtered_sentence)   

mystring = ' '.join(filtered_sentence)

doc = nlp(mystring)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.is_stop)

cols = ("text", "stopword")
rows = []

for t in doc:
    row = [t.text,  t.is_stop]
    rows.append(row)

df = pd.DataFrame(rows, columns=cols)
    
df