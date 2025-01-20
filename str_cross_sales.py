__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pandas as pd
from itertools import combinations
import streamlit as st
import re
import time
from datetime import date
import os
import json
import base64
import requests
from datetime import datetime
import math
import altair as alt
from concurrent.futures import ProcessPoolExecutor, as_completed
import gspread
from google.oauth2.service_account import Credentials
from vis_context_tools import promt_chooser,screenshot_by_url, get_text_content_by_url, llm_analysis_of_image, llm_analysis_of_text
from get_clickup_info_by_client import get_ab_test_tickets_info_by_client_name, get_list_of_clients_names
import hmac
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import RunReportRequest, Metric, Dimension
from openai import OpenAI

TOOLS_LIST = ["Main page", "Cross-Sales App", "VisContext Analyzer", "GA4 Chat"]
DA_NAMES = ["Amar","Djordje","Tarik","Axel","Denis","JDK","other"]



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.

if "CLIENTS_LIST" not in st.session_state.keys() or st.session_state["CLIENTS_LIST"] == None:
    st.session_state['CLIENTS_LIST'] = get_list_of_clients_names()
if "GA4 CHAT TEXT" not in st.session_state.keys() or st.session_state["GA4 CHAT TEXT"] == None:
    st.session_state['GA4 CHAT TEXT'] = ""
def update_contact():
    st.session_state.contact = st.session_state.contact_select

def send_slack_notification(data):
    # Замените на свой Webhook-URL
    webhook_url = st.secrets['SLACK_WEBHOOK_URL']

    # Форматируем сообщение
    message = f"""
    :bell: **New request from {data['contact']}:**
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
        print("Notification of successful sending to Slack")
    else:
        print(f"Error sending notification: {response.status_code}, {response.text}")


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


def interpret_query(user_input,property_id):
    client = OpenAI()
    client_ga = BetaAnalyticsDataClient()
    metadata = client_ga.get_metadata(name=f"properties/{property_id}/metadata")
    metrics = [m.api_name for m in metadata.metrics]
    dimensions = [d.api_name for d in metadata.dimensions]
    print(dimensions)
    print(metrics)
    system_prompt = (
        """You are the assistant who interprets any user requests and converts them into JSON parameters for the Google Analytics Data API.
If the user asks about metrics, dimensions, specify them explicitly.
Example: 'If he ask give him number of Active users in the last week' -> {'metrics': ['name':'activeUsers'], 'date_ranges': [{'start_date': '7daysAgo', 'end_date': 'today'}]}"""
f"""Available metrics:
{metrics}
Available dimensions:
{dimensions}
IMPORTANT! YOUR answer must contain only JSON parameters for the Google Analytics Data API and nothing else!"""
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.2
    )
    pattern = r'\{.*\}'
    text = response.choices[0].message.content
    print(response.choices[0].message.content)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        extracted = match.group()
        print("Извлечённая часть:", extracted)
    else:
        print("Фигурные скобки не найдены")

    query_params = json.loads(extracted)

    print(query_params)
    request = RunReportRequest(
                property=f"properties/{property_id}",
                metrics=query_params['metrics'],
                dimensions=query_params.get('dimensions', []),
                date_ranges=query_params['date_ranges']
            )
    print(client_ga.run_report(request))
    return client_ga.run_report(request)


if __name__ == '__main__':
    st.set_page_config(
        page_title="DA toolkit",
        page_icon="acc_big_logo.png",  # Путь к favicon
        layout="wide",
    )
    st.image("acc_big_logo.png",width=100)
    main_page, tab2, tab3, tab4 = st.tabs(TOOLS_LIST)

    with main_page:
        st.header("Welcome to the Toolkit!")
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
                DA_NAMES,

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
        if 'cs_df' not in st.session_state.keys():
            st.session_state['cs_df'] = None

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
            st.session_state['cs_df'] = None
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

            st.success("Processing complete!")
            st.session_state['cs_df'] = cs_df



        if type(st.session_state['cs_df']) != None:
            st.write(st.session_state['cs_df'])
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(st.session_state['cs_df'])

            st.download_button(
                label="Download result in CSV",
                data=csv_data,
                file_name="cross_sales_results.csv",
                mime="text/csv"
            )
            sort_option = st.selectbox(
                "Sort chart by:",
                ["Transactions (Descending)", "Transactions (Ascending)", "AOV (Descending)", "AOV (Ascending)"],
            )
            if sort_option == "Transactions (Descending)":
                top_items = st.session_state['cs_df'].sort_values(by="transactions", ascending=False)
            elif sort_option == "Transactions (Ascending)":
                top_items = st.session_state['cs_df'].sort_values(by="transactions", ascending=True)
            elif sort_option == "AOV (Descending)":
                top_items = st.session_state['cs_df'].sort_values(by="AOV", ascending=False)
            else:
                top_items = st.session_state['cs_df'].sort_values(by="AOV", ascending=True)

            num_items = st.slider("Number of combinations to display:", 5,20, 200)
            top_items = top_items.head(num_items)

            top_items['item_comb'] = top_items['item_comb'].apply(lambda x: ', '.join(x))
            combined_chart = alt.Chart(top_items).mark_bar().encode(
                y=alt.Y('item_comb', title='Product Combinations',axis=alt.Axis( labelLimit=150),sort=None),
                x=alt.X('transactions', title='Number of Transactions'),
                color=alt.Color('AOV', title='AOV'),
                tooltip=['item_comb', 'transactions', 'AOV']
            ).interactive()


            st.altair_chart(combined_chart, use_container_width=True)
    with tab3:


        st.header("VisContext Analyzer")
        analysis_type = st.selectbox(
            "Select the type of analysis",
            ["Choose","Text Content Analysis", "Full Screenshot Analysis", "Section-Specific Screenshot Analysis"]
        )
        result_type = st.selectbox(
            "Select the type of analysis result",
            ["Choose","Strengths and Weaknesses Analysis", "Hypothesis Formation for A/B Testing"]
        )

        if analysis_type != 'Choose' and result_type != 'Choose':
            company_name = st.selectbox(
            "Choose the company name",
            st.session_state['CLIENTS_LIST']

        )
            target_audience = st.text_input("Describe your target audience")
            current_date = date.today()
            product_type = st.text_input("Describe the product sold by the company")
            industry_type = st.text_input("Write the industry in which this company operates.")
            page_type = st.selectbox(
            "Select the type of page to analyze",
            ["Choose", "Homepage","Product Page","Colection Page","Cart page", "Checkout Page", "CartLayer","Navigation bar","Navigation Layer", "Landing page"]
        )
            promt_chooser_radio = st.radio("Select whose prompt template to use",['Andrii', 'Denis'])
            if analysis_type == "Text Content Analysis":
                url_of_text_content = st.text_input("Insert a link to the page from which the text content will be taken")
            elif analysis_type == "Full Screenshot Analysis":
                url_or_img_of_content = st.radio("Select a method for taking a screenshot",["Upload screenshot","Take from page URL"])
                if url_or_img_of_content=="Upload screenshot":
                    img_of_content = st.file_uploader("Load screenshot",type=["jpg","png"])
                elif url_or_img_of_content=="Take from page URL":
                    url_of_img_content = st.text_input(
                        "Insert a link to the page from which the screenshot will be taken")
            elif analysis_type == "Section-Specific Screenshot Analysis":
                img_of_content = st.file_uploader("Load screenshot of page section",type=["jpg","png"])

            if st.button("Generate prompt"):




                if company_name == "":
                    st.error("Error! The company name field cannot be empty.")
                    st.stop()
                elif target_audience == "":
                    st.error("Error! The target audience field cannot be empty.")
                    st.stop()
                elif product_type == "":
                    st.error("Error! The product type field cannot be empty.")
                    st.stop()
                elif page_type == "Choose":
                    st.error("Error! You must select an analysis page.")
                    st.stop()
                elif 'url_of_text_content' in locals() and url_of_text_content =='':
                    st.error("Error! You must provide the URL to the page with the tag for analysis.")
                    st.stop()
                elif 'img_of_content' in locals() and img_of_content ==None:
                    st.error("Error! You must provide a screenshot of the page being analyzed..")
                    st.stop()
                elif 'url_of_img_content' in locals() and url_of_img_content =='':
                    st.error("You must provide the page URL for visual analysis.")
                    st.stop()


                prompt_ex_res_list = promt_chooser(analysis_type = analysis_type,
                                                   result_type=result_type,
                                                   company_name = company_name,
                                                   target_audience = target_audience,
                                                   current_date = current_date,
                                                   product_type = product_type,
                                                   page_type = page_type,
                                                   industry_type = industry_type,
                                                   promt_chooser_radio = promt_chooser_radio)

                st.session_state["fixed_prompt"] = prompt_ex_res_list[0]
                st.session_state["fixed_result"] = prompt_ex_res_list[1]


                if result_type == "Hypothesis Formation for A/B Testing":
                    st.session_state['ab_tests_of_client'] = get_ab_test_tickets_info_by_client_name(company_name)
                    st.session_state["fixed_prompt"] = prompt_ex_res_list[0] + st.session_state['ab_tests_of_client']

            st.divider()
            st.subheader("Prompt for LLM analysis")
            fixed_prompt = st.text_area(label="This prompt will be used as a task for LLM analysis. If you want to change the task, change the text below.",
                                        value=st.session_state.get("fixed_prompt", ""),
                                        height=300,
                                        key="fixed_prompt")
            st.divider()
            st.subheader("Expected output of LLM analysis")
            fixed_result = st.text_area(label="This prompt will be used as a hint about the expected type of result for LLM analysis. If you want to change the expected result, change the text below. IMPORTANT! If you use Text Content Analysis, do not change the last line 'Some Text Content', later during the analysis it will be replaced by the text context from the web page!",
                                        value=st.session_state.get("fixed_result", ""),
                                        height=300,
                                        key="fixed_result")

            if st.button("Start LLM analysis"):

                with st.spinner("Analyzing..."):

                    # Пример долгой задачи

                    fixed_prompt = st.session_state["fixed_prompt"]
                    fixed_result = st.session_state["fixed_result"]
                    
                    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
                    os.environ["OPENAI_ORGANIZATION"] = st.secrets['OPENAI_ORGANIZATION']
                    if analysis_type == "Text Content Analysis":
                        text_content = get_text_content_by_url(url_of_text_content)

                        result_of_llm_analysis = llm_analysis_of_text(text_content,fixed_prompt,fixed_result)
                    elif analysis_type == "Full Screenshot Analysis" :
                        if url_or_img_of_content =="Upload screenshot":
                            screenshot = img_of_content.read()

                            screenshot = base64.b64encode(screenshot).decode("utf-8")
                        elif url_or_img_of_content =="Take from page URL":
                            screenshot = screenshot_by_url(url_of_img_content)
                        result_of_llm_analysis = llm_analysis_of_image(screenshot,f"{fixed_prompt}\n\nGive your final answer in the following structure:\n{fixed_result}")
                    elif analysis_type == "Section-Specific Screenshot Analysis":
                        screenshot = img_of_content.read()
                        screenshot = base64.b64encode(screenshot).decode("utf-8")
                        result_of_llm_analysis = llm_analysis_of_image(screenshot,f"{fixed_prompt}\n\nGive your final answer in the following structure:\n{fixed_result}")
                st.success("Analysis complete!")
                st.write(result_of_llm_analysis)
    with tab4:

        st.title("GA4 Chat")
        ga4_client_name = st.selectbox(
            "Choose the company name",
            st.session_state['CLIENTS_LIST']
        )
        ga4_chat_or_table = st.radio("Select whether you want to create a Chat with GA4 data or create a Table.",["Chat","Table"])
        ga4_text_for_prompt = st.text_area(label="Enter a question or task for your GA4 data.",
                                        height=300)
        if st.button("Get an Answer"):
            st.session_state["GA4 CHAT TEXT"] += ga4_text_for_prompt
            property_id = st.secrets["GA4_DICT_CLIENT_PROPERTY_ID"][0][ga4_client_name]
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/macbook/PycharmProjects/ACC_Cross_Sales/Key.json"

            response = interpret_query(ga4_text_for_prompt,property_id)
            # Пример преобразования строки JSON в Python-объект

            # Форматирование ответа для отображения
            for row in response.rows:
                dimension_value = row.dimension_values[0].value if row.dimension_values else None
                metric_value = row.metric_values[0].value if row.metric_values else None
                st.write(f"{dimension_value}: {metric_value}")
        if st.button("Clear Chat"):
            st.session_state["GA4 CHAT TEXT"] = ""

        st.write(st.session_state["GA4 CHAT TEXT"])








