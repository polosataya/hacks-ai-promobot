import telebot
import pandas as pd
import json
from annoy import AnnoyIndex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import emoji
import re
import nltk
import spacy
from spacy import displacy


bot = telebot.TeleBot('Тут_должен_быть_токен')

#===============================================================================
# Функции для згрузки
#===============================================================================

def load_vectorizer(filename):
    tfidf = TfidfVectorizer()
    tfidf = joblib.load(pathModel+filename)
    return tfidf

def load_logreg(filename):
    logit = LogisticRegression()
    logit = joblib.load(pathModel+filename)
    return logit

def load_stopwords():
    # подготовка моделей обработки текста
    nltk.download('stopwords')
    stopwords_nltk = nltk.corpus.stopwords.words('russian') #лист русский стоп-слов
    stopwords_nltk_en = nltk.corpus.stopwords.words('english')
    stopwords_nltk.extend(stopwords_nltk_en) #часть текста на английском
    new_stop = ['здравствовать', 'подсказать', 'сказать', "пожалуйста", "спасибо",  "благодарить", "извинить",
            'вопрос','тема', "ответ", "ответить", "почему", "что",
            'которая', 'которой', 'которую', 'которые', 'который', 'которых', 'это', "мочь",
            'вообще', "всё", "весь", "ещё", "просто",  "якобы", "причём", 'точно', "хотя", "именно", 'неужели',
             "г", "ул", "город", "улица"]
    stopwords_nltk.extend(new_stop)
    return stopwords_nltk

def load_nlp():
    nlp = spacy.load('ru_core_news_md')
    return nlp

#===============================================================================
# Работа с текстом
#===============================================================================

def full_clean(text):
    '''очистка строки текста'''
    text = emoji.demojize(text)
    text=re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9#]", " ", text)
    text = text.lower()
    text = re.sub(" +", " ", text) #оставляем только 1 пробел
    text = text.replace("добрый день", "").replace("добрый вечер", "").replace("доброе утро", "").replace("сообщение без текста", "").replace("да подтверждаю", "").replace("до сих пор", "")
    text = text.strip()
    #токены для моделей
    tokens = [token.lemma_ for token in nlp(text) if token.lemma_ not in stopwords_nltk]
    #для tfidf на вход текст
    text = " ".join(tokens)
    return text

def classify_data(logit, data):
    # предсказание логистической регрессией для вектора предложения
    return logit.predict(data)[0]

def predict_main(text_input):
    try:
        text_clean = full_clean(text_input)
        tfidf_embed=tfidf.transform([text_clean])

        # Предсказание группы тем
        predict_group = classify_data(logit_group, tfidf_embed)
        # Предсказание темы
        predict_title = classify_data(logit_title, tfidf_embed)
        # Предсказание исполнителя
        predict_otdel = classify_data(logit_otdel, tfidf_embed)

        result_text = f"*Группа тем:* {predict_group}\n*Тема:* {predict_title}\n*Исполнитель:* {predict_otdel}"

        return result_text

    except:
        return "Не могу ответить на вопрос"

#=========================================================================================
# Загрузки моделей и файлов
#=========================================================================================

pathModel="model/"

tfidf = load_vectorizer('tfidf.pkl')
logit_group = load_logreg("logit_group.sav")
logit_title = load_logreg("logit_title.sav")
logit_otdel = load_logreg("logit_otdel.sav")
stopwords_nltk = load_stopwords()
nlp = load_nlp()

#=========================================================================================
# Бот
#=========================================================================================

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "/start":
        bot.send_message(message.from_user.id, "Привет, этот бот поможет классифицировать ваше сообщение.")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши в сообщении вопрос.")
    else:
        # строка, которую вводит пользователь
        result = predict_main(message.text)
        bot.send_message(message.from_user.id, result, parse_mode="Markdown")

bot.polling(none_stop=True, interval=0)
