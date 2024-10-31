import pandas as pd
from itertools import combinations
import streamlit as st
import re
import time
import json
import requests
from datetime import datetime
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import gspread
from google.oauth2.service_account import Credentials

TOOLS_LIST = ["Main page", "Cross-Sales App", "Tool number 2", "Tool number 3"]
DA_NAMES = ["Amar","Djordje","Tarik","other"]
def update_contact():
    st.session_state.contact = st.session_state.contact_select

def send_slack_notification(data):
    # Замените на свой Webhook-URL
    webhook_url = "https://hooks.slack.com/services/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"

    # Форматируем сообщение
    message = f"""
    :bell: **Новое обращение от пользователя:**
    *Topic:* {data['topic']}
    *Tool:* {data['tool_name']}
    *Description:* {data['description']}
    *Priority:* {data['priority']}
    *Contact:* {data['contact']}
    *Date and time:* {data['submit_time']}
    """

    # Создаем JSON-полезную нагрузку
    payload = {"text": message}

    # Отправляем POST-запрос на Webhook URL
    response = requests.post(webhook_url, json=payload)

    # Проверка ответа
    if response.status_code == 200:
        print("Уведомление успешно отправлено в Slack")
    else:
        print(f"Ошибка при отправке уведомления: {response.status_code}, {response.text}")


def count_cross_sells(row, product_matrix, full_df):
    try:
        # Убедимся, что row содержит только названия продуктов
        items = list(row['item_comb'])  # Извлекаем первые два элемента комбинации
        mask = product_matrix[items].all(axis=1)
        filtered_df = full_df[mask]
        transactions = len(filtered_df)
        aov = filtered_df['total_revenue'].sum()
        return pd.Series([row['item_comb'], transactions, aov], index=['item_comb', 'transactions', 'AOV'])
    except KeyError as e:
        print(f"Ошибка: {e}")
        return pd.Series([row['item_comb'], 0, 0], index=['item_comb', 'transactions', 'AOV'])


# Разбиваем DataFrame на чанки
def split_dataframe(df, chunk_size):
    num_chunks = math.ceil(len(df) / chunk_size)
    for i in range(num_chunks):
        yield df.iloc[i * chunk_size:(i + 1) * chunk_size]


# Параллельная обработка
def process_chunk(chunk, product_matrix, full_df):
    return chunk.apply(lambda row: count_cross_sells(row, product_matrix, full_df), axis=1)


def process_in_parallel(df, chunk_size, product_matrix, full_df):
    results = []
    futures = []
    total_chunks = math.ceil(len(df) / chunk_size)

    with ProcessPoolExecutor(max_workers=10) as executor:
        for chunk in split_dataframe(df, chunk_size):
            futures.append(executor.submit(process_chunk, chunk, product_matrix, full_df))

        for i, future in enumerate(as_completed(futures)):
            result = future.result()

            results.append(result)

            # Обновляем прогресс-бар
            progress = (i + 1) / total_chunks
            progress_bar.progress(min(progress, 1.0))
    return pd.concat(results, ignore_index=True)

if __name__ == '__main__':
    st.set_page_config(
        page_title="DA toolkit",
        page_icon="acc_big_logo.png",  # Путь к favicon
        layout="wide",
    )
    st.image("acc_big_logo.png",width=100)
    main_page, tab2, tab3, tab4 = st.tabs(TOOLS_LIST)

    with main_page:
        st.header("Welcome to the DAs Toolkit!")
        st.write('''This platform is designed to provide small but powerful tools to make your daily tasks easier and more efficient. We're just getting started, and there’s more to come!Have ideas for tools that could streamline your work? We’d love to hear your suggestions—your input will help shape future updates. Let’s build a smarter, more efficient workplace together!''')
        st.write(
            "If you have ideas for a new tool, suggestions for improvement, or problems, please fill out the form below."
        )

        # Форма для ввода данных
        with st.form(key='feedback_form'):
            topic = st.selectbox(
                "Select the subject of your request",
                ["New idea for a tool", "Modify an existing tool", "Report a problem/question"]
            )
            tool_name = st.text_input("Name of the tool (if applicable)")

            description = st.text_area("Description of the idea / problem", height=200)

            priority = st.selectbox("Priority", ["Low", "Medium", "High"])

            contact = st.selectbox(
                "Your name",
                DA_NAMES
            )




            submit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Время подачи заявки

            # Кнопка отправки формы
            submit_button = st.form_submit_button(label="Send")

        # Обработка отправки формы
        if submit_button:

            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
                     "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

            creds = Credentials.from_service_account_info(dict(st.secrets['GS_DS_CRED']), scopes=scope)


            client = gspread.authorize(creds)
            sheet = client.open("DAs toolkit requests").sheet1  # или client.open_by_key("sheet_id")

            if description.strip() == "":
                st.error("Description cannot be empty.")
            else:
                data = {
                    "topic": topic,
                    "tool_name": tool_name,
                    "description": description,
                    "priority": priority,
                    "contact": contact,
                    "submit_time": submit_time
                }
                sheet.append_row(list(data.values()))
                send_slack_notification(data)# Добавляем строку в Google Sheets
                st.success("Спасибо! Ваша форма успешно отправлена.")

    with tab2:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        df_main = pd.DataFrame()
        if uploaded_file:


            df_main = pd.read_csv(uploaded_file)
            st.write("File uploaded successfully:")
            st.write(df_main.head())
            if len(df_main.columns) < 4:
                st.error("The downloaded CSV file does not have enough columns. Make sure the file contains 4 columns:\n'transaction_id', 'product_name', 'product_quantity', 'total_revenue'")
            elif len(df_main.columns) > 4:
                st.error("The downloaded CSV file has more columns than needed. Make sure the file contains 4 columns:'transaction_id', 'product_name', 'product_quantity', 'total_revenue'")

            if list(df_main.columns) != ['transaction_id', 'product_name', 'product_quantity', 'total_revenue']:
                df_main.columns = ['transaction_id', 'product_name', 'product_quantity', 'total_revenue']
        st.subheader("Select options")
        radio_quantity_items = st.radio("Choose quantity of items for calculations in a cross-selling table", ["Choose",1,2,3,4,5])
        radio_table_type_var = st.radio("Choose type of result table", ['None','25 items','100 items','25 AOV','100 AOV'])
        radio_aov_filter = st.radio("Choose range of items purchse AOV in a cross-selling table", ['None','<50','50-75','75-100','>100'])
        radio_item_quantity_filter = st.radio("Select the minimum purchase quantity for each item", ['None',5,10,25,50])

        use_custom_pattern = st.checkbox("Use a regular expression pattern to remove variations from product names")

        if use_custom_pattern:
            re_pattern = st.text_input("Enter RegEx pattern", value="", max_chars=50)
        else:
            re_pattern = None  # Если чекбокс не установлен, шаблон пуст


        if st.button("Start processing"):

            # Проверка, что выбраны корректные параметры
            if radio_quantity_items == "Choose":
                st.error("Please select the number of items using the radio button")
            elif not uploaded_file:
                st.error("Please upload the CSV file.")
            else:
                # Начинаем обработку только если всё заполнено
                st.success("Processing started...")
            progress_bar = st.progress(0)
            progress_text = st.empty()


            if re_pattern !=None:
                df_main['product_name'] = df_main['product_name'].apply(lambda x : re.sub(re_pattern, '', x))


            valid_transaction_ids = df_main['transaction_id'].value_counts()[
                df_main['transaction_id'].value_counts() >= radio_quantity_items
                ].index
            df_main = df_main[df_main['transaction_id'].isin(valid_transaction_ids)]

            if radio_item_quantity_filter !='None':
                valid_product_names = df_main.groupby('product_name')['product_quantity'].sum()[
                    df_main.groupby('product_name')['product_quantity'].sum() >= radio_item_quantity_filter
                    ].index
                df_main = df_main[df_main['product_name'].isin(valid_product_names)]





            # Create df transaction_id|product_name
            item_df = df_main.groupby(['transaction_id'])['product_name'].apply('#'.join).reset_index()

            # Create df transaction_id|quantity|total_revenue
            int_df = df_main.groupby(['transaction_id'])[['product_quantity', 'total_revenue']].sum().reset_index()

            # Add to int_df column of quantity uniques items
            item_df['quantity_uniques'] = item_df['product_name'].str.count("#") + 1

            # Concenate to dataframes to get full values
            full_df = pd.merge(item_df, int_df, on="transaction_id")

            # Create Series of names of uniques items
            products = pd.Series(df_main['product_name'].unique())

            # Create finall dataframe
            cs_df = pd.DataFrame({"item_comb": list(combinations(products, radio_quantity_items)), 'transactions': 0, 'AOV' : 0})


            current_progress_value = 0.0
            # Create function for counting and summing amount of transactions and AOV contais tuple of products
            product_matrix = full_df['product_name'].str.get_dummies(sep='#')


            #start_time = time.time()
            cs_df = process_in_parallel(cs_df, chunk_size=100, product_matrix=product_matrix, full_df=full_df)
            #end_time = time.time()
            #execution_time = end_time - start_time
            #print(f"Время выполнения старой функции: {execution_time:.2f} секунд")


            # Apply to all lines in cs_df count_cross_sells function and fill 'transactions' column init



            # Remove lones with 0 value
            cs_df = cs_df[cs_df['transactions'] != 0]

            # Calculate real AOV
            cs_df['AOV'] = cs_df['AOV'] / cs_df['transactions']

            # Clean data to good view
            cs_df['transactions'] = cs_df['transactions'].astype(int)
            cs_df['AOV'] = cs_df['AOV'].round(2)
            cs_df = cs_df.sort_values(by=['transactions'], ascending=False)

            # Filter cross-salles dataframe by cs_df_finall
            if radio_aov_filter!=None:
                if radio_aov_filter == '<50':
                    cs_df = cs_df[cs_df['AOV'] < 50]
                elif radio_aov_filter == '50-75':
                    cs_df = cs_df[(50 <= cs_df['AOV']) & (cs_df['AOV'] < 75)]
                elif radio_aov_filter == '75-100':
                    cs_df = cs_df[(75 <= cs_df['AOV']) & (cs_df['AOV'] < 100)]
                elif radio_aov_filter == '>100':
                    cs_df = cs_df[cs_df['AOV']>= 100]
                else:
                    cs_df = cs_df

            # Filter cross-salles table type
            if radio_table_type_var != None:
                if radio_table_type_var == '25 items':
                    cs_df = cs_df.sort_values(by=['transactions'], ascending=False).iloc[0:25]
                elif radio_table_type_var == '100 items':
                    cs_df = cs_df.sort_values(by=['transactions'], ascending=False).iloc[0:100]
                elif radio_table_type_var == '25 AOV':
                    cs_df = cs_df.sort_values(by=['AOV'], ascending=False).iloc[0:25]
                elif radio_table_type_var == '100 AOV':
                    cs_df = cs_df.sort_values(by=['AOV'], ascending=False).iloc[0:100]
                else:
                    cs_df = cs_df

            st.write(cs_df)
            st.success("Processing complete!")
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')


            csv_data = convert_df_to_csv(cs_df)

            st.download_button(
                label="Download result in CSV",
                data=csv_data,
                file_name="cross_sales_results.csv",
                mime="text/csv"
            )
