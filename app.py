import hydralit as hy
import streamlit as st
import pandas as pd
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import emoji
import re
import nltk
import spacy
from spacy import displacy

st.set_page_config( page_title="Обработка обращений граждан",
                    page_icon="🤖", layout="wide", initial_sidebar_state="expanded",
                    menu_items={'Get Help': None,'Report a bug': None,'About': None})

hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#============================================================================
# Функции для загрузки
#============================================================================

pathModel="model/"

@st.cache_resource
def load_vectorizer(filename):
    '''Функция для загрузки модели векторизатора'''
    tfidf = TfidfVectorizer()
    tfidf = joblib.load(pathModel+filename)
    return tfidf

@st.cache_resource
def load_logreg(filename):
    '''Функция для загрузки модели классификатора'''
    logit = LogisticRegression()
    logit = joblib.load(pathModel+filename)
    return logit

@st.cache_data
def load_stopwords():
    ''' подготовка моделей обработки текста '''
    nltk.download('stopwords')
    stopwords_nltk = nltk.corpus.stopwords.words('russian') #лист русских стоп-слов
    stopwords_nltk_en = nltk.corpus.stopwords.words('english')
    stopwords_nltk.extend(stopwords_nltk_en) #часть текста на английском
    new_stop = ['здравствовать', 'подсказать', 'сказать', "пожалуйста", "спасибо",  "благодарить", "извинить",
                'вопрос', 'тема', "ответ", "ответить", "почему", "что",
                'которая', 'которой', 'которую', 'которые', 'который', 'которых', 'это', "мочь",
                'вообще', "всё", "весь", "ещё", "просто",  "якобы", "причём", 'точно', "хотя", "именно", 'неужели',
                "г", "ул", "город", "улица"]  #специфичные стоп-слов
    stopwords_nltk.extend(new_stop)
    return stopwords_nltk
@st.cache_resource
def load_nlp():
    nlp = spacy.load('ru_core_news_md')
    return nlp

@st.cache_data
def load_upload(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file, sep=',')
    except:
        data = pd.read_csv(uploaded_file, sep=';')
    return data

#============================================================================
# Функции для работы с тектом
#============================================================================
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
    return text, tokens

def preprocess_text(df):
    '''подготовка текста к подаче в модель колонкой'''
    new_corpus = []
    new_tokens = []

    for text in df:
        text, tokens = full_clean(text)
        new_corpus.append(text)
        new_tokens.append(tokens)
    return new_corpus, new_tokens

def tfidf_embeding(model=None, df=None):
    '''Преобразование текста в мешок слов'''
    X = model.transform(df)
    return X.toarray()

# Функция для классификации данных
def classify_data(logit, data):
    ''' предсказание логистической регрессией для вектора предложения '''
    return logit.predict(data)

#============================================================================
# Загружаем модели
#============================================================================

tfidf = load_vectorizer('tfidf.pkl')
logit_group = load_logreg("logit_group.sav")
logit_title = load_logreg("logit_title.sav")
logit_otdel = load_logreg("logit_otdel.sav")
stopwords_nltk = load_stopwords()
nlp = load_nlp()

#============================================================================
# Приложение
#============================================================================

# Создаем главное приложение Hydralit
app = hy.HydraApp(title='Многостраничное приложение Streamlit')

# Главная страница для ввода данных и их классификации
@app.addapp('Ввод')
def input_and_classify_page():
    st.title("Ввод и классификация данных")

    # Поле ввода для текста
    text_input = st.text_area('Введите текст для классификации')

    # Кнопка для классификации данных
    if st.button('Классифицировать'):
        # преобразование текста в вектор
        text_clean, _ = full_clean(text_input)
        tfidf_embed=tfidf.transform([text_clean])

        # Предсказание группы тем
        predict_group = classify_data(logit_group, tfidf_embed)[0]
        # Предсказание темы
        predict_title = classify_data(logit_title, tfidf_embed)[0]
        # Предсказание исполнителя
        predict_otdel = classify_data(logit_otdel, tfidf_embed)[0]

        # Отображение результатов
        st.subheader("Результаты классификации:")

        # Вывод значений из первой строки CSV файла
        st.write(f"Группа тем: {predict_group}")
        st.write(f"Тема: {predict_title}")
        st.write(f" ")
        st.write(f"Исполнитель: {predict_otdel}")

        # NER
        st.subheader("Текст с выделенными именованными сущностями:")
        doc = nlp(text_input)
        ent_html = displacy.render(doc, style="ent", jupyter = False)

        st.write(f" ")
        # Display the entity visualization in the browser:
        st.markdown(ent_html, unsafe_allow_html=True)

# Функция для загрузки файла и отображения результатов
@app.addapp('Загрузка')
def upload_and_display_page():
    st.title("Загрузка файла и отображение результатов")

    # Создание кнопки загрузки файла
    uploaded_file = st.file_uploader("Выберите файл для загрузки", type=["csv"])

    # Кнопка для классификации загруженных данных
    if uploaded_file is not None and st.button('Классифицировать'):
        # Загрузка файла
        uploaded_data = load_upload(uploaded_file)

        uploaded_data['text_clean'], uploaded_data['tokens'] = preprocess_text(uploaded_data['Текст инцидента'])
        tfidf_embed = tfidf_embeding(model=tfidf, df=uploaded_data['text_clean'])

        # Классификация данных
        # Предсказание группы тем
        predict_group = classify_data(logit_group, tfidf_embed)
        # Предсказание темы
        predict_title = classify_data(logit_title, tfidf_embed)
        # Предсказание исполнителя
        predict_otdel = classify_data(logit_otdel, tfidf_embed)

        #Заполнение таблицы
        uploaded_data["Группа тем"]=predict_group
        uploaded_data["Тема"] = predict_title
        uploaded_data["Исполнитель"] = predict_otdel

        # Отображение всей таблицы
        st.subheader("Результаты классификации:")
        st.dataframe(uploaded_data[["Текст инцидента", "Группа тем", "Тема", "Исполнитель"]])

        # Кнопка для скачивания
        csv = uploaded_data[["Текст инцидента", "Группа тем", "Тема", "Исполнитель"]].to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="result.csv"><button>Скачать результат</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# Запускаем приложение Hydralit
app.run()