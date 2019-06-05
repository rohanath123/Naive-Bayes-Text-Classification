import sklearn 
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline 

# USES SKLEARN'S PIPELINE TO MAKE A CONTINOUS SET OF SUCCESSIVELY OCCURING FUNCTIONS, COUNTVECTORIZER() TO TOKENIZE AND BUILD A SET OF VOCABULARY, AND TFIDFTRANSFORMER TO CONVERT WORDS 
# TO VECTORS
# USES SKLEARN'S MULTINOMIALNB (MULTINOMIAL NAIVE BAYES) ALGORITHM. GOOGLE JUST "NAIVE BAYES". MULTINOMIAL JUST MEANS MORE THAN 2 CLASS CLASSIFICATION USING NAIVE BAYES.	
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])