
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


# get_ipython().magic(u'pylab inline')


# In[3]:


df = pd.read_csv('fake_or_real_news.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df = df.set_index('Unnamed: 0')


# In[7]:


df.head()


# In[8]:


y = df.label


# In[9]:


df = df.drop('label', axis=1)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)

# -*- coding: utf-8 -*-
from googletrans import Translator
translator = Translator()
# translator.translate('안녕하세요')
p = translator.translate('अमेरिकी')
print p.text



# In[ ]:


new_news = translator.translate('अमेरिकी विदेश मंत्री ने अपने उत्तर कोरियाई समकक्ष के साथ परमाणु निरस्त्रीकरण की योजना के क्रियान्वयन पर गंभीर बातचीत की.  विदेश मंत्री पोम्पिओ उत्तर कोरियाई नेता किम जोंग - उन के विश्वासपात्र किम जोंग चोल से दूसरे दिन बातचीत के लिए प्योंगयांग के एक आलीशान गेस्ट हाउस में मौजूद थे. पोम्पिओ का यह तीसरा प्योगयांग दौरा है , जिसपर उनके उत्तर कोरियाई समकक्ष किम जोंग चोल ने मजाक में कहा कि उन्हें अब शायद इस शहर की आदत होने लगी है. उन्होंने कहा , ‘हम जितना मिलेंगे, उतनी हमारी दोस्ती गहरी होती जाएगी. आज की बैठक काफी सकारात्मक रही. इस पर पोम्पिओ ने कहा, हां, मैं इससे सहमत हूं. गौरतलब है कि बीते जून के महीने में ही अमेरिका के राष्ट्रपति डोनाल्ड ट्रंप और उत्तर कोरिया के तानाशाह किम जोंग उन के बीच शिखर वार्ता हुई थी. इससे पहले दोनों ही नेता एक दूसरे को देख लेने की धमकी दे रहे थे.')
# new_news = translator.translate('दुनिया की सबसे बड़ी खोजी संस्था नासा ने एक हैरान कर देने वाले रिसर्च से ये पता लगाया है कि फेसबुक, इंस्टाग्राम, व्हाट्सएप्प और ट्वीटर के बाहर भी जीवन सम्भव है. जिसके बाद दुनियाभर के लोगों में हैरानगी का माहौल है. उन्हें यकीन नहीं हो रहा है कि ऐसा भी हो सकता है!अधिक जानकारी के लिए जब हमने उस खोजी वैज्ञानिक दल के एक सदस्य साइंटिस्ट ‘विलियम जेम्स सदाशिव फर्नांडीज’ से बात की तो उन्होंने बताया कि “हमने दिन रात एक करके ये रिसर्च किया कि क्या फेसबुक व्हाट्सएप्प वगैरह से बाहर भी जीवन है? अगर हाँ तो कैसे दिखते होंगे वो लोग जो इन चीजों के बिना भी जी रहे हैं. इसी रिसर्च में हमने चार साल लगा दिए और आख़िरकार हमें सफलता मिल ही गई. हमने खोज निकाला कि इन सबसे बाहर भी जीवन है. और वहां के लोग हंसी ख़ुशी रह रहे हैं।जब से नासा ने ये ऐलान किया है कि दुनिया भर के सोशल मीडियाजीवी ऐसे लोगों को देखने के लिए बहुत उत्साहित हैं जो फेसबुक व्हाट्सएप्प के बिना जी रहे हैं. वो उनके साथ सेल्फ़ी लेकर व्हाट्सएप्प पर लगाना चाहते हैं।वहीं नासा की इस अभूतपूर्व कामयाबी के बाद डोनाल्ड ट्रम्प ने ‘ट्वीट करके‘ उन्हें उनकी कामयाबी के लिए बधाई दी है।')
new_news=new_news.text
print "news ",new_news


# In[ ]:


count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
#rand_art = count_vectorizer.transform(['In a last dash, final hail mary attempt to end a Donald Trump run for the White House once and for all, the National Review has decided to eviscerate the Republican front runner on the basis that he is not a conservative.\n\nIt will not work.\n\nPublications like National Review, run by elite conservatives have given us George W. Bush and his wars, No Child Left Behind, Medicare Part D, huge deficits caused by Republican consultants spending to woo select voters, Mitt Romneycare Romney, John McCain...the list goes on and on.\n\nWilliam F. Buckley, who founded National Review, used the magazine to publish a stellar series of essays by conservative intellectuals who helped foster the Reagan Revolution.\n\nSince then, movement conservatism has not been a powerful enough force to make things better for the working classes in the country.\n\nThis vacuum, created by the conservative elites who have backed RINOs (Republicans in Name Only) and candidates who are antithetical to conservatism, is what created the opportunity for Donald Trump to rise.In fact, publications like National Review have such a blind spot, they never even saw devout, pro-America nationalists like Trump taking off.They arent credible in their criticism of Trump because they never saw it coming.Beyond that, and most importantly, they told us we -- the conservatives who are sick and tired of elistist, establishment blunders -- were wrong.And they still dont get it.Trumps appeal stretches far beyond disgruntled, outside the country club conservatives. His potential for crossover support, especially with blue collar and working class voters, is huge. Most establishment Republicans have never met a blue collar worker (unless they were fixing their Jacuzzi).I can see Trump winning coal miners, unionized construction workers, auto workers, steel workers, Teamsters, etc.Trump may even score a larger share of black votes with his immigration stand. His appeal to working class voters is a very under reported story, but its evident because even President Barack Obama himself mentioned Trump by name during an interview with NPR in which he said that Trump is tapping into the anger of the blue collar white male.This showcases just how scared the left is when it comes to Trumps potential to tear into demographics that Democrats have largely considered theirs.The bed wetters at the RNC are dreaming of a GOP that grows because it attracts Latinos, pro-abortion millennial women and other hopelessly Democratic voters. Trumps coalition of adding working class voters (who actually work) makes more sense.I have respect for National Review as an institution, but the cover and series of articles designed to hurt Trump only hurts the elitest, Beltway crowd they represent because it exposes why he is the seemingly solid and unstoppable frontrunner: its because of them.They have failed us, not Trump. Donald Trump is merely capitalizing on a moment in a pursuit to make America Great Again, in spite of the failures of the conservative movement.Just like they were too dense to see Trumps rise, they dont understand why it occurred.National Review, its time for your Man in the Mirror moment. People are more concerned about the country they love, than they are your brand of conservatism.By trying to take out the most popular candidate in this race who has the best general election shot of any of them to win the White House and reverse the progressive policies of Barack Obama, Beltway, frat boy type elitists are proving my point: they dont get it.And from the looks of it, they never will.'])
# new_news = 'In a last dash, final hail mary attempt to end a Donald Trump run for the White House once and for all, the National Review has decided to eviscerate the Republican front runner on the basis that he is not a conservative.\n\nIt will not work.\n\nPublications like National Review, run by elite conservatives have given us George W. Bush and his wars, No Child Left Behind, Medicare Part D, huge deficits caused by Republican consultants spending to woo select voters, Mitt Romneycare Romney, John McCain...the list goes on and on.\n\nWilliam F. Buckley, who founded National Review, used the magazine to publish a stellar series of essays by conservative intellectuals who helped foster the Reagan Revolution.\n\nSince then, movement conservatism has not been a powerful enough force to make things better for the working classes in the country.\n\nThis vacuum, created by the conservative elites who have backed RINOs (Republicans in Name Only) and candidates who are antithetical to conservatism, is what created the opportunity for Donald Trump to rise.In fact, publications like National Review have such a blind spot, they never even saw devout, pro-America nationalists like Trump taking off.They arent credible in their criticism of Trump because they never saw it coming.Beyond that, and most importantly, they told us we -- the conservatives who are sick and tired of elistist, establishment blunders -- were wrong.And they still dont get it.Trumps appeal stretches far beyond disgruntled, outside the country club conservatives. His potential for crossover support, especially with blue collar and working class voters, is huge. Most establishment Republicans have never met a blue collar worker (unless they were fixing their Jacuzzi).I can see Trump winning coal miners, unionized construction workers, auto workers, steel workers, Teamsters, etc.Trump may even score a larger share of black votes with his immigration stand. His appeal to working class voters is a very under reported story, but its evident because even President Barack Obama himself mentioned Trump by name during an interview with NPR in which he said that Trump is tapping into the anger of the blue collar white male.This showcases just how scared the left is when it comes to Trumps potential to tear into demographics that Democrats have largely considered theirs.The bed wetters at the RNC are dreaming of a GOP that grows because it attracts Latinos, pro-abortion millennial women and other hopelessly Democratic voters. Trumps coalition of adding working class voters (who actually work) makes more sense.I have respect for National Review as an institution, but the cover and series of articles designed to hurt Trump only hurts the elitest, Beltway crowd they represent because it exposes why he is the seemingly solid and unstoppable frontrunner: its because of them.They have failed us, not Trump. Donald Trump is merely capitalizing on a moment in a pursuit to make America Great Again, in spite of the failures of the conservative movement.Just like they were too dense to see Trumps rise, they dont understand why it occurred.National Review, its time for your Man in the Mirror moment. People are more concerned about the country they love, than they are your brand of conservatism.By trying to take out the most popular candidate in this race who has the best general election shot of any of them to win the White House and reverse the progressive policies of Barack Obama, Beltway, frat boy type elitists are proving my point: they dont get it.And from the looks of it, they never will.'


# In[ ]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
rand_news = tfidf_vectorizer.transform([new_news])


# In[ ]:


tfidf_vectorizer.get_feature_names()[-10:]


# In[ ]:


# count_vectorizer.get_feature_names()[:10]


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


clf = MultinomialNB() 


# In[ ]:


clf.fit(tfidf_train, y_train)
predi = clf.predict(rand_news)


# In[ ]:

print "answer is ",predi


# In[ ]:


clf.fit(tfidf_train, y_train)
pred1 = clf.predict(tfidf_test[18])


# In[ ]:


pred1


# In[ ]:


X_test.iloc[18]


# In[ ]:


y_test.iloc[18]


# In[ ]:


clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm,normalize=True, classes=['FAKE', 'REAL'])


# In[ ]:


clf = MultinomialNB(alpha=0.5)


# In[ ]:


clf.fit(count_train, y_train)
pred = clf.predict(count_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm,normalize=True, classes=['FAKE', 'REAL'])


# And indeed, with absolutely no parameter tuning, your count vectorized training set `count_train` is visibly outperforming your TF-IDF vectors!

# In[ ]:


linear_clf = PassiveAggressiveClassifier(n_iter=50)


# In[ ]:


linear_clf.fit(tfidf_train, y_train)
pred = linear_clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm,normalize=True, classes=['FAKE', 'REAL'])


# Wow! 
# 
# I'm impressed. The confusion matrix looks different and the model classifies our fake news a bit better. We can test if tuning the `alpha` value for a `MultinomialNB` creates comparable results. You can also use [parameter tuning with grid search](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#parameter-tuning-using-grid-search) for a more exhaustive search.

# In[ ]:


clf = MultinomialNB(alpha=0.01)


# In[ ]:


last_score = 0
for alpha in np.arange(0,1,.1):
    nb_classifier = MultinomialNB(alpha=alpha)
    nb_classifier.fit(tfidf_train, y_train)
    pred = nb_classifier.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    if score > last_score:
        clf = nb_classifier
    print("Alpha: {:.2f} Score: {:.5f}".format(alpha, score))


# Not quite... At this point, it might be interesting to perform parameter tuning across all of the classifiers, or take a look at some other [scikit-learn Bayesian classifiers](http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes). You could also test with a Support Vector Machine (SVM) to see if that outperforms the Passive Aggressive classifier.
# 
# But I am a bit more curious about what the Passive Aggressive model actually *has* learned. So let's move onto introspection.

# In[ ]:


def most_informative_feature_for_binary_classification(vectorizer, classifier, n=100):
    """
    See: https://stackoverflow.com/a/26980472
    
    Identify most important features if given a vectorizer and binary classifier. Set n to the number
    of weighted features you would like to show. (Note: current implementation merely prints and does not 
    return top classes.)
    """

    class_labels = classifier.classes_
    feature_names = vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]
    topn_class2 = sorted(zip(classifier.coef_[0], feature_names))[-n:]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)

    print()

    for coef, feat in reversed(topn_class2):
        print(class_labels[1], coef, feat)


most_informative_feature_for_binary_classification(tfidf_vectorizer, linear_clf, n=30)


# You can also do this in a pretty obvious way with only a few lines of Python, by zipping your coefficients to your features and taking a look at the top and bottom of your list.

# In[ ]:


feature_names = tfidf_vectorizer.get_feature_names()


# In[ ]:


### Most real
sorted(zip(clf.coef_[0], feature_names), reverse=True)[:20]


# In[ ]:


### Most fake
sorted(zip(clf.coef_[0], feature_names))[:20]


# In[ ]:


tokens_with_weights = sorted(list(zip(feature_names, clf.coef_[0])))


# In[ ]:


hash_vectorizer = HashingVectorizer(stop_words='english', non_negative=True)
hash_train = hash_vectorizer.fit_transform(X_train)
hash_test = hash_vectorizer.transform(X_test)


# In[ ]:


clf = MultinomialNB(alpha=.01)


# In[ ]:


clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, normalize=True, classes=['FAKE', 'REAL'])


# In[ ]:


clf = PassiveAggressiveClassifier(n_iter=5)


# In[ ]:


clf.fit(hash_train, y_train)
pred = clf.predict(hash_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
plot_confusion_matrix(cm, normalize=True, classes=['FAKE', 'REAL'])


# In[ ]:


y.shape,df.shape

