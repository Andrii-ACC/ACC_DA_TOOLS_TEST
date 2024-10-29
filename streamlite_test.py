import streamlit as st

st.title("Calculate of sales")

# Ввод данных пользователем
price = st.number_input("Product price", min_value=0.0, step=0.01)
quantity = st.number_input("Amounbt of sales", min_value=0)

# Вычисление результата
total = price * quantity

# Кнопка для запуска вычислений
if st.button("Calculate"):
    st.success(f"Total amount: {total} $.")
uploaded_file = st.file_uploader("Load CSV file with sales", type=["csv"])
