# coding: utf-8
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print twenty_train.target_names #prints all the categories
print("\n".join(twenty_train.data[0].split("\n")[:3])) #prints first line of the first data file
# print "bmnbmn ",twenty_train.data[:3]
# print "bmnbmn ",twenty_train.target[:]
print twenty_train.target.max()

i=0;
# from googletrans import Translator
# translator = Translator()
# new_news = translator.translate('अमेरिका के केंसास सिटी में एक 26 वर्षीय युवक की गोली मारकर हत्या कर दी गई. स्थानीय पुलिस के अनुसार मृतक युवक भारतीय है और तेलंगाना राज्य का रहने वाला है. पूरी घटना शुक्रवार शाम की है. घटना उस वक्त हुई जब पीड़ित युवक एक रेस्त्तरां में खाना खाने गया था. पुलिस के अनुसार मृतक युवक की पहचान सार्थक कोपू के रूप में की गई है. वह अमेरिका की एक यूनिवर्सिटी में पढ़ाई कर रहा था. पुलिस के अनुसार घटना के बाद घायल छात्र को पास के अस्पताल ले जाया गया था जहां डॉक्टरों ने उसे मृत घोषित कर दिया है.')
# new_news = new_news.text
new_news = "America is a city of criminals."
print new_news

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(twenty_train.data)
import pickle

pickle.dump(count_vect, open("count_vect.pickle", "wb"))

jago = count_vect.transform(['From: irwin@cmptrc.lonestar.org (Irwin Arnstein)\nSubject: Re: Recommendation on Duc\nSummary: What\'s it worth?\nDistribution: usa\nExpires: Sat, 1 May 1993 05:00:00 GMT\nOrganization: CompuTrac Inc., Richardson TX\nKeywords: Ducati, GTS, How much? \nLines: 13\n\nI have a line on a Ducati 900GTS 1978 model with 17k on the clock.  Runs\nvery well, paint is the bronze/brown/orange faded out, leaks a bit of oil\nand pops out of 1st with hard accel.  The shop will fix trans and oil \nleak.  They sold the bike to the 1 and only owner.  They want $3495, and\nI am thinking more like $3K.  Any opinions out there?  Please email me.\nThanks.  It would be a nice stable mate to the Beemer.  Then I\'ll get\na jap bike and call myself Axis Motors!\n\n-- \n-----------------------------------------------------------------------\n"Tuba" (Irwin)      "I honk therefore I am"     CompuTrac-Richardson,Tx\nirwin@cmptrc.lonestar.org    DoD #0826          (R75/6)\n-----------------------------------------------------------------------\n'])
print X_train_counts.shape
print i
i+=1 #0

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print X_train_tfidf
pickle.dump(tfidf_transformer, open("tfidf.pickle", "wb"))

mago = tfidf_transformer.transform(jago)
print X_train_tfidf.shape

print i
i+=1  #1

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.01).fit(X_train_tfidf, twenty_train.target)
#dump
pickle.dump(clf, open("clf.pickle", "wb"))
print i
i+=1 #2
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))), ('tfidf', TfidfTransformer(use_idf=True)), ('clf', MultinomialNB(alpha=0.01))])
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

pickle.dump(text_clf, open("text_clf.pickle", "wb"))

print i
i+=1 #3

import numpy as np
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted = text_clf.predict(twenty_test.data)
pred3 = text_clf.predict(['An international footballer and a model who won the right to have a lawful humanist wedding in Northern Ireland have failed to secure a judicial declaration for a wider law change after judges said existing legislation meant it was not needed. The appeal court judges in the case of the model Laura Lacole and the Leeds United and Republic of Ireland footballer Eunan O’Kane, ruled against a wider law change on humanist marriages but said existing laws already avoided discrimination.The three judges found the current law enables couples to apply for temporary authorisation for a humanist celebrant to conduct marriages. For this reason, the court in Belfast said the definition of legal marriages did not need to be expanded to incorporate specific “beliefs” such as humanist.Despite losing the substantive case, Lacole welcomed the finding as a “step forward”, insisting a precedent had been set for couples to have legally recognised humanist weddings in the region.The couple originally went to court because they were refused temporary authorisation from Northern Ireland’s General Register Office (GRO) when they applied. They won the right to have their ceremony legally recognised.In the couple’s original high court victory last year, the trial judge had ruled that the law should be altered to include the term “or beliefs” among definitions of religious ceremonies.An appeal by Northern Ireland’s attorney general, John Larkin, Stormont’s Department of Finance and the GRO was allowed by the appeal court judges on Thursday. The granting of that authorisation was not appealed against. Humanism is a non-religious belief system that rejects the concepts of a higher deity or afterlife, believing humans steer their own destiny. Humanist marriages are already legally recognised in Scotland, but not in England and Wales. They are also recognised in the Republic of Ireland.Lacole said while the law would not be changed, the consequences of the appeal court ruling meant couples now had a clear pathway to have lawful humanist ceremonies. “I am really happy,” she said outside the court of appeal in Belfast. “The fact we are now walking away having the door be opened for the non-religious in Northern Ireland to have a humanist ceremony befitting of their beliefs is amazing. “So other people can ultimately have the wedding we had, and that was the goal – so we are really happy.” The couple’s lawyer, Ciaran Moynagh, also drew positives from the judgment. “Essentially it does open the door today that humanist ceremonies will be recognised in Northern Ireland,” he said“The initial ask was that ‘belief’ was read into the legislation or the legislation was declared incompatible and that would have made the government seriously consider the marriage law and look at rewriting it or changing provisions in it.'])

print np.mean(predicted == twenty_test.target)
print twenty_train.target_names[pred3[0]]
print i
i+=1 #4

from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
pred4 = text_clf.predict(['A motorcycle, often called a bike, motorbike, or cycle, is a two- or three-wheeled motor vehicle.[1] Motorcycle design varies greatly to suit a range of different purposes: long distance travel, commuting, cruising, sport including racing, and off-road riding. Motorcycling is riding a motorcycle and related social activity such as joining a motorcycle club and attending motorcycle rallies.'])
print np.mean(predicted_svm == twenty_test.target)
print twenty_train.target_names[pred4[0]]
print i
i+=1 #5

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
main = gs_clf
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

filename = 'finalized_classification_model.sav'
pickle.dump(gs_clf, open(filename, 'wb'))

print i
i+=1 #6
pred5 = gs_clf.predict([new_news])
print twenty_train.target_names[pred5[0]]
print gs_clf.best_params_
print main.best_score_

from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
tum = gs_clf_svm
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)
print gs_clf_svm.best_params_
print tum.best_score_
print i
i+=1 #7

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])

print i
i+=1 #8

import nltk
nltk.download()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
   def build_analyzer(self):
       analyzer = super(StemmedCountVectorizer, self).build_analyzer()
       return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                      ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)

predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

print np.mean(predicted_mnb_stemmed == twenty_test.target)
print "end"
