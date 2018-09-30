# coding: utf-8
import pickle
import sys
import random
import time

name = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']
# print "model loading started"
ticks = time.time()
filename = 'gs_clf.sav'
loaded_model = pickle.load(open(filename, 'rb'))
# print time.time()-ticks
# print "model loaded!"

from googletrans import Translator
translator = Translator()
# p = translator.translate('अमेरिकी')
# print p.text

# new_news = translator.translate('अमेरिकी विदेश मंत्री ने अपने उत्तर कोरियाई समकक्ष के साथ परमाणु निरस्त्रीकरण की योजना के क्रियान्वयन पर गंभीर बातचीत की.  विदेश मंत्री पोम्पिओ उत्तर कोरियाई नेता किम जोंग - उन के विश्वासपात्र किम जोंग चोल से दूसरे दिन बातचीत के लिए प्योंगयांग के एक आलीशान गेस्ट हाउस में मौजूद थे. पोम्पिओ का यह तीसरा प्योगयांग दौरा है , जिसपर उनके उत्तर कोरियाई समकक्ष किम जोंग चोल ने मजाक में कहा कि उन्हें अब शायद इस शहर की आदत होने लगी है. उन्होंने कहा , ‘हम जितना मिलेंगे, उतनी हमारी दोस्ती गहरी होती जाएगी. आज की बैठक काफी सकारात्मक रही. इस पर पोम्पिओ ने कहा, हां, मैं इससे सहमत हूं. गौरतलब है कि बीते जून के महीने में ही अमेरिका के राष्ट्रपति डोनाल्ड ट्रंप और उत्तर कोरिया के तानाशाह किम जोंग उन के बीच शिखर वार्ता हुई थी. इससे पहले दोनों ही नेता एक दूसरे को देख लेने की धमकी दे रहे थे.')
# # new_news = translator.translate('दुनिया की सबसे बड़ी खोजी संस्था नासा ने एक हैरान कर देने वाले रिसर्च से ये पता लगाया है कि फेसबुक, इंस्टाग्राम, व्हाट्सएप्प और ट्वीटर के बाहर भी जीवन सम्भव है. जिसके बाद दुनियाभर के लोगों में हैरानगी का माहौल है. उन्हें यकीन नहीं हो रहा है कि ऐसा भी हो सकता है!अधिक जानकारी के लिए जब हमने उस खोजी वैज्ञानिक दल के एक सदस्य साइंटिस्ट ‘विलियम जेम्स सदाशिव फर्नांडीज’ से बात की तो उन्होंने बताया कि “हमने दिन रात एक करके ये रिसर्च किया कि क्या फेसबुक व्हाट्सएप्प वगैरह से बाहर भी जीवन है? अगर हाँ तो कैसे दिखते होंगे वो लोग जो इन चीजों के बिना भी जी रहे हैं. इसी रिसर्च में हमने चार साल लगा दिए और आख़िरकार हमें सफलता मिल ही गई. हमने खोज निकाला कि इन सबसे बाहर भी जीवन है. और वहां के लोग हंसी ख़ुशी रह रहे हैं।जब से नासा ने ये ऐलान किया है कि दुनिया भर के सोशल मीडियाजीवी ऐसे लोगों को देखने के लिए बहुत उत्साहित हैं जो फेसबुक व्हाट्सएप्प के बिना जी रहे हैं. वो उनके साथ सेल्फ़ी लेकर व्हाट्सएप्प पर लगाना चाहते हैं।वहीं नासा की इस अभूतपूर्व कामयाबी के बाद डोनाल्ड ट्रम्प ने ‘ट्वीट करके‘ उन्हें उनकी कामयाबी के लिए बधाई दी है।')
# new_news=new_news.text
# print "news ",new_news

# # tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# rand_news = loaded_vectorizer.transform([new_news])
# pred = loaded_model.predict(rand_news)
# print pred


# print "news is ",sys.argv[1]
new_news = sys.argv[1]
new_news = translator.translate(new_news)
new_news=new_news.text
# print new_news
# rand_news = loaded_vectorizer.transform([new_news])
pred = loaded_model.predict([new_news])
print (name[pred[0]])
