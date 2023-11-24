import hydralit as hy
import streamlit as st
import pandas as pd
import base64

st.set_page_config(
    page_title="Обработка обращений граждан",
    page_icon="🤖", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'Get Help': None,'Report a bug': None,'About': None})

hide_streamlit_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Функция для классификации данных
def classify_data(data):
    # Ваш код для классификации данных
    # Здесь может быть вызов вашей модели или другой обработки данных
    # Пока что я вставил заглушку, которая возвращает тот же DataFrame
    return data

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
        # Загрузка CSV файла (заглушка для примера)
        df = pd.read_csv('data/temp.csv')

        # Функция для классификации данных
        def classify_data(input_data):
            # Ваш код для классификации данных
            # Пока что используем заглушку, возвращая первую строку из CSV файла
            return df.iloc[0]

        # Здесь вы можете вызвать функцию классификации
        result_data = classify_data(text_input)

        # Отображение результатов
        st.subheader("Результаты классификации:")

        # Вывод значений из первой строки CSV файла
        st.write(f"Группа тем: {result_data['Группа тем']}")
        st.write(f"Тема: {result_data['Тема']}")
        st.write(f"Исполнитель: {result_data['Исполнитель']}")

# Функция для загрузки файла и отображения результатов
@app.addapp('Загрузка')
def upload_and_display_page():
    st.title("Загрузка файла и отображение результатов")

    # Создание кнопки загрузки файла
    uploaded_file = st.file_uploader("Выберите файл для загрузки", type=["csv"])

    # Кнопка для классификации загруженных данных
    if uploaded_file is not None and st.button('Классифицировать'):
        # Загрузка файла
        uploaded_data = pd.read_csv(uploaded_file)

        # Классификация данных
        classified_data = classify_data(uploaded_data)

        # Отображение всей таблицы
        st.subheader("Результаты классификации:")
        st.write(classified_data)

        # Кнопка для скачивания
        csv = classified_data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="result.csv"><button>Скачать результат</button></a>'
        st.markdown(href, unsafe_allow_html=True)

# Запускаем приложение Hydralit
app.run()