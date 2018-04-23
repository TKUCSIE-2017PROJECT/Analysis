# coding=UTF-8
import jieba
import json
import codecs

import sys
import os
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import feature_extraction 
 
import numpy as np
import scipy

 
jieba.set_dictionary('dict.txt.big')
jieba.load_userdict('繁體中文詞庫.txt')

f = open('stop_words.txt', 'r',encoding="utf-8")
stopwords = []  
for line in f.readlines():
    stopwords.append(line.strip()) 
f.close() 



Makeup = ['Dcard_Json\\makeup_dcard_1.json']
Travel = ['Dcard_Json\\travel_dcard_1.json','Taiwannews_Json\\travel_Taiwannews_1.json']
Finance =['Dcard_Json\\money_dcard_1.json','Appledaily_Json\\finance_appledaily_1.json',
          'Taiwannews_Json\\finance_Taiwannews_1.json']
Sport = ['Dcard_Json\\sport_dcard_1.json','Appledaily_Json\\sport_appledaily_1.json']
TVepisode = ['Dcard_Json\\tvepisode_dcard_1.json','Appledaily_Json\\entertainment_appledaily_1.json']
_3C = ['Dcard_Json\\3c_dcard_1.json','Appledaily_Json\\3C_appledaily_1.json'
       ,'Taiwannews_Json\\technology_Taiwannews_1.json']

ACG = ['Dcard_Json\\acg_dcard_1.json']
Game = ['Dcard_Json\\game_dcard_1.json']
Vehicle =['Dcard_Json\\vehicle_dcard_1.json']
Movie =['Dcard_Json\\movie_dcard_1.json']
Boy = ['Dcard_Json\\boy_dcard_1.json']
Girl = ['Dcard_Json\\girl_dcard_1.json']
Food = ['Dcard_Json\\food_dcard_1.json']

Talk = ['Dcard_Json\\talk_dcard_1.json']
Internation = ['Appledaily_Json\\international_appledaily_1.json'
               ,'Taiwannews_Json\\internation_Taiwannews_1.json']
Shopping = ['Dcard_Json\\buyonline_dcard_1.json']


data_name = {'Makeup':Makeup,
             'Travel':Travel,
             'Finance':Finance,
             'Sport':Sport,
             'TVepisode':TVepisode,
             '3C':_3C ,
             'ACG':ACG,
             'Game':Game,
             'Vehicle':Vehicle,
             'Movie':Movie,
             'Boy':Boy,
             'Girl':Girl,
             'Food':Food,
             'Talk':Talk,
            'Internation':Internation,
             'Shopping':Shopping}
 

topic = ['Makeup','Travel','Finance','Sport','TVepisode','3C',
                'ACG','Game','Vehicle','Movie','Boy','Girl','Food','Talk',
                 'Shopping','Internation']

for kind in range(len(topic)):
    
    content_cut = []
    content_title = []
    content_excerpt = []
    print(topic[kind],kind)
    qqq = 0
    qqqq = 0
    
    for address in range(len(data_name[topic[kind]])):
        print(data_name[topic[kind]][address],address)   
        file = open('jsoncrawler\\'+data_name[topic[kind]][address], 'r', encoding='utf-8')
        f = file.read()
        jsondata = json.loads(f)
        file.close()
        
        for content in jsondata:
            content_str = ''
            word_str = ''
            train = []
            try:
                content_title.append(content['title']) 
                content_excerpt.append(content['excerpt'])
                line = jieba.cut(content['title']+content['excerpt'])
                train.append([w for w in line if w not in stopwords])
                qqqq += 1 
            except :
                content_title.append(content['topic']) 
                content_excerpt.append(content['content'])
                
                if '報導）' in content['content']:
                    a = content['content'].find('報導）')
                    content_change = content['content'][a+3:]
                else:
                    content_change = content['content']
                    
                line = jieba.cut(content['topic']+content_change)
                train.append([w for w in line if w not in stopwords])
                qqq += 1 
        
            for word_1 in train[0]:
                if word_1 == ' ':
                    continue
                word_str = word_str+' ' + word_1
            content_cut.append(word_str)
    print(qqqq,qqq)
#將文字傳化成陣列
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(content_cut)
    analyze = vectorizer.build_analyzer()
    weight = X.toarray()  


    print (len(weight))  
    print (weight[:5,:5])
    
    import lda  
    import lda.datasets
    model = lda.LDA(n_topics=4, n_iter=500, random_state=1)  
    model.fit(np.asarray(weight))     # model.fit_transform(X) is also available  
    topic_word = model.topic_word_ 

#文档-主题（Document-Topic）分布 
    
    doc_topic = model.doc_topic_  
    #print("type(doc_topic): {}".format(type(doc_topic)))  
    #print("shape: {}".format(doc_topic.shape))
    #输出前10篇文章最可能的Topic  
    label = [] 
    topic0_count = 0
    topic1_count = 0
    topic2_count = 0
    topic3_count = 0
    for n in range(len(weight)):  
        topic_most_pr = doc_topic[n].argmax()  
        label.append(topic_most_pr)  
        if topic_most_pr == 0:
            topic0_count += 1
        elif topic_most_pr == 1:
            topic1_count += 1
        elif topic_most_pr == 2:
            topic2_count += 1
        elif topic_most_pr == 3:
            topic3_count += 1
        #print("doc: {} topic: {}".format(n, topic_most_pr))
    
    
#输出主题中的TopN关键词  
    word = vectorizer.get_feature_names()  

    #print (topic_word[:, :5])  
    n = 5   
    for i, topic_dist in enumerate(topic_word):    
        topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]    
        #print(u'*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
        if i == 0:
            topic0 =  ' '.join(topic_words)
        elif i == 1:
            topic1 =  ' '.join(topic_words)
        elif i == 2:
            topic2 =  ' '.join(topic_words)
        elif i == 3:
            topic3 =  ' '.join(topic_words)
    
    all_list = []
    for t in range(len(label)):
        content_dict = {
            'topic' : int(label[t]),
            'title' : content_title[t],
            'excerpt' :content_excerpt[t]
        } 
        
        all_list.append(content_dict)
   
    topic_dict = {
        'topic0':topic0,
        'topic1':topic1,
        'topic2':topic2,
        'topic3':topic3,
        'topic0_count':topic0_count,
        'topic1_count':topic1_count,
        'topic2_count':topic2_count,
        'topic3_count':topic3_count
    }
    all_dict = {
        'all_content' : all_list,
        'topic_dict' : topic_dict
    }
    
    with codecs.open('json_LDA\\'+ topic[kind] +'.json', 'w', 'utf-8') as file2:
        json.dump(all_dict, file2,indent = 5,sort_keys=True,ensure_ascii=False)
    file2.close()