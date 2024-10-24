import streamlit as st

st.title("Подсчет продаж")

# Ввод данных пользователем
price = st.number_input("Цена товара", min_value=0.0, step=0.01)
quantity = st.number_input("Количество проданных единиц", min_value=0)

# Вычисление результата
total = price * quantity

# Кнопка для запуска вычислений
if st.button("Рассчитать"):
    st.success(f"Общая сумма продаж: {total} grn.")