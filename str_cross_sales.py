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
from google.api_core.exceptions import PermissionDenied
from base_models import GA4_Chat_Answer
import time
from itertools import combinations
import swifter
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

TOOLS_LIST = ["Main page", "Cross-Sales App", "VisContext Analyzer", "GA4 Chat"]
DA_NAMES = ["Amar","Djordje","Tarik","Axel","Denis","JDK","other"]



def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True

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
if "ga4_result_meta_data" not in st.session_state:
    st.session_state["ga4_result_meta_data"] = {}
if "ga4_result_table" not in st.session_state:
    st.session_state["ga4_result_table"] = pd.DataFrame()
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



if __name__ == '__main__':
    st.set_page_config(
        page_title="DA toolkit",
        page_icon="acc_big_logo.png",  # Путь к favicon
        layout="wide",
    )
    st.image("acc_big_logo.png",width=100)
    main_page, tab2, tab3, tab4 = st.tabs(TOOLS_LIST)

    with main_page:
        print("Tab Main opened!")
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
        print("Tab 2 opened!")
        if 'cs_df' not in st.session_state:
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
            # Начинаем обработку только если всё заполнено

            # progress_bar = st.progress(0) old
            # progress_text = st.empty() old


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
            print(len(df_main))






            # OLD
            # # Create df transaction_id|product_name
            # item_df = df_main.groupby(['transaction_id'])['product_name'].apply('#'.join).reset_index()
            #
            # # Create df transaction_id|quantity|total_revenue
            # int_df = df_main.groupby(['transaction_id'])[['product_quantity', 'total_revenue']].sum().reset_index()
            #
            # # Add to int_df column of quantity uniques items
            # item_df['quantity_uniques'] = item_df['product_name'].str.count("#") + 1
            #
            # # Concenate to dataframes to get full values
            # full_df = pd.merge(item_df, int_df, on="transaction_id")
            #
            # # Create Series of names of uniques items
            # products = pd.Series(df_main['product_name'].unique())
            #
            # # Create finall dataframe
            # cs_df = pd.DataFrame({"item_comb": list(combinations(products, radio_quantity_items)), 'transactions': 0, 'AOV' : 0})



            # Create function for counting and summing amount of transactions and AOV contais tuple of products
            # product_matrix = full_df['product_name'].str.get_dummies(sep='#')
            #
            # #start_time = time.time()
            # cs_df = process_in_parallel(cs_df, chunk_size=100, product_matrix=product_matrix, full_df=full_df)
            # ________________________
            # OLD
            agg_df = df_main.groupby('transaction_id').agg({
                'product_name': lambda x: '#'.join(x),
                'product_quantity': 'sum',
                'total_revenue': 'sum'
            }).reset_index()

            agg_df['quantity_uniques'] = agg_df['product_name'].str.count("#") + 1



            def get_combinations(products, size):
                return list(combinations(sorted(products.split('#')), size))


            col1, col2, col3 = st.columns(3)
            with col1:
                with st.spinner("Processing started! Creating a list of items combinations for each transaction..."):

                    start_time = time.time()
                    agg_df['item_comb'] = agg_df['product_name'].swifter.apply(lambda x: get_combinations(x, radio_quantity_items))
                    end_time = time.time()
                    execution_time = end_time - start_time
                st.success(f'Creating a list of items combinations for each transaction Completed successfully! The process took: {round(execution_time,3)} seconds')

            with col2:
                with st.spinner("Exploding of combination lists..."):
                    start_time = time.time()
                    agg_df = agg_df.explode('item_comb')
                    end_time = time.time()
                    execution_time = end_time - start_time
                st.success(f'Exploding of combination lists was successful!The process took: {round(execution_time,3)} seconds')
            with col3:
                with st.spinner("Forming the final table..."):
                    start_time = time.time()
                    cs_df = agg_df.groupby('item_comb').agg(
                        transactions=('transaction_id', 'count'),
                        AOV=('total_revenue', 'mean')
                    ).reset_index()
                    end_time = time.time()
                    execution_time = end_time - start_time
                st.success(f'The final table was formed successfully! The process took: {round(execution_time,3)} seconds')

            # ________________________
            
            # Remove lones with 0 value
            cs_df = cs_df[cs_df['transactions'] != 0]

            # Calculate real AOV

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



        if type(st.session_state['cs_df']) != type(None):
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
        print("Tab 3 opened!")

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
                    
                    os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY_Vizcontext_Tool']
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

        os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY_GA4_Chat']
        os.environ["OPENAI_ORGANIZATION"] = st.secrets['OPENAI_ORGANIZATION']

        client_ga = BetaAnalyticsDataClient.from_service_account_info(st.secrets["ACC_GC_KEY"])

        dict_clientname_prop = st.secrets["GA4_DICT_CLIENT_PROPERTY_ID"]
        if "ga4_result_meta_data" not in st.session_state:
            st.session_state["ga4_result_meta_data"] = {}
        if "metrics_list" not in st.session_state:
            st.session_state["metrics_list"] = []
        if "dimensions_list" not in st.session_state:
            st.session_state["dimensions_list"] = []
        st.title("GA4 Chat")
        ga4_client_name = st.selectbox(
            "Choose the company name",
            dict_clientname_prop.keys(),

        )
        property_id = dict_clientname_prop[ga4_client_name]

        if "ga4_client_name" not in st.session_state or ga4_client_name != st.session_state['ga4_client_name']:
            st.session_state['ga4_client_name'] = ga4_client_name
            st.session_state['ga4_result_meta_data'] = {}
            try:
                metadata = client_ga.get_metadata(name=f"properties/{property_id}/metadata")
                st.session_state['metrics_list'] = [m.api_name for m in metadata.metrics]
                st.session_state['dimensions_list'] = [d.api_name for d in metadata.dimensions]
            except PermissionDenied as e:
                st.error("The application does not have sufficient permissions for this client property.")
                st.stop()
        ga4_model = st.radio("Select the GPT model that will be used to process your request.",["gpt-4o-mini","gpt-4o"])
        col1, col2, col3, col4 = st.columns(4)
        # В первой колонке выбираем метрики
        with col1:
            metric_multiselect = st.multiselect("Select Metric", options=st.session_state['metrics_list'],max_selections=10)

        # Во второй колонке выбираем измерения
        with col2:
            dimension_multiselect = st.multiselect("Select Dimension", options=st.session_state['dimensions_list'], max_selections=9)
        with col3:
            selected_dates = st.date_input(
                "Choose date range",
                value=(date.today(), date.today()),  # Значения по умолчанию (сегодня)
                min_value=date(2000, 1, 1),  # Минимальная доступная дата
                max_value=date.today()  # Максимальная доступная дата (сегодня)
            )





        ga4_text_for_prompt = st.text_area(label="Enter a question or task for your GA4 data.",height=300)
        create_timeline_graph_check = st.checkbox("Do you want to plot a line chart on a timeline for this data? Warning! The Date dimension will be forcibly included, avoid metrics and dimensions that are incompatible with this dimension.")
        print(create_timeline_graph_check)
        if st.button("Get an Answer") :

            if len(ga4_text_for_prompt)  >= 2:

                property_id = dict_clientname_prop[ga4_client_name]


                if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                    start_date, end_date = selected_dates
                    ga4_text_for_prompt = f"{ga4_text_for_prompt}\nTake data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.\n"

                if len(metric_multiselect)> 0 and len(dimension_multiselect) > 0:
                    ga4_text_for_prompt = f"{ga4_text_for_prompt}\nAdditionally, try to include these Metrics: \n{metric_multiselect}\n and These Dimensions \n{dimension_multiselect}\n in your request."
                elif len(metric_multiselect)>0:
                    ga4_text_for_prompt = f"{ga4_text_for_prompt}\nAdditionally, try to include these Metrics: \n{metric_multiselect} in your request."
                elif len(dimension_multiselect) > 0:
                    ga4_text_for_prompt = f"{ga4_text_for_prompt}\nAdditionally, try to include these Dimensions: \n{dimension_multiselect} in your request."

                if create_timeline_graph_check:
                    ga4_text_for_prompt = f"{ga4_text_for_prompt}\nAlso, be sure to break down the data by the Date dimension!"
                st.session_state['ga4_text_for_prompt'] = ga4_text_for_prompt


                agent = GA4_Chat_Answer(client_ga, ai_model = 'gpt-4o-mini',ga4_property=property_id )
                with st.spinner("The AI Agent is in the process of generating a data table..."):
                    response = agent.answer(ga4_text_for_prompt)

                    st.session_state['ga4_result_raw'] = response[1]
                    st.session_state['ga4_result_api'] = response[2]
                    for col in response[2]['metrics']:
                        col = col['name']
                        response[0][col] = (
                            pd.to_numeric(response[0][col], errors='coerce')
                            .fillna(0)
                        )
                    for col_names in response[0]:
                        if response[0][col_names].dtype == 'float64':
                            response[0][col_names] = response[0][col_names].round(2)

                    st.session_state['ga4_result_table'] = response[0]

            elif len(ga4_text_for_prompt)  < 2:
                st.error("""Prompt can't be so short""")


        if 'ga4_result_api' in st.session_state:
            if create_timeline_graph_check and 'date' in st.session_state['ga4_result_table'].columns:
                print(st.session_state['ga4_result_api'])
                if len(st.session_state['ga4_result_api']['dimensions'])>0:
                    df = st.session_state['ga4_result_table'].loc[:, st.session_state['ga4_result_table'].columns != 'date']
                    summary = (
                        df.groupby([dim['name'] for dim in st.session_state['ga4_result_api']['dimensions'] if dim['name'] != 'date'], as_index=False)[[metr['name'] for metr in st.session_state['ga4_result_api']['metrics']]]
                        .sum()
                        .reset_index(drop=True)
                    )
                else:
                    # Без измерений — просто одна строка со всеми метриками
                    summary = pd.DataFrame([st.session_state['ga4_result_table'][[metr['name'] for metr in st.session_state['ga4_result_api']['metrics']]].sum().to_dict()])
                    print(summary)
                st.dataframe(summary.sort_values(by=[[metr['name'] for metr in st.session_state['ga4_result_api']['metrics']]][0],ascending=False))

            elif create_timeline_graph_check and st.session_state['ga4_result_table'].index.name == 'date':
                df = st.session_state['ga4_result_table'].reset_index()
                st.write(df.loc[:, df.columns != 'date'])
            else:
                st.write(st.session_state['ga4_result_table'])

            with st.expander("Show main API parameters"):
                st.json(st.session_state['ga4_result_api'])

        if st.button("Check API"):
            property_id = dict_clientname_prop[ga4_client_name]
            if 'ga4_result_raw' not in st.session_state:
                st.error("""To correctly check the quality of the generated API request, it is necessary to "Get an answer".""")
                st.stop()
            agent = GA4_Chat_Answer(client_ga, ai_model='gpt-4o-mini', ga4_property=property_id )
            with st.spinner("The AI Agent is in the process of checking API..."):
                response_api_check = agent.check_api(st.session_state['ga4_text_for_prompt'], st.session_state['ga4_result_raw'])
            st.session_state['ga4_result_meta_data'] = response_api_check

        if st.session_state['ga4_result_meta_data'].get('overall_verdict',[]) == "Fully compliant":
            st.success("Fully compliant")
            st.write(f"Note: {st.session_state['ga4_result_meta_data']['note']}")
        elif st.session_state['ga4_result_meta_data'].get('overall_verdict',[]) == "Some differences":
            st.warning("Some differences")
            st.write(f"Differences: {st.session_state['ga4_result_meta_data']['differences']}")
            st.write(f"Note: {st.session_state['ga4_result_meta_data']['note']}")
        elif st.session_state['ga4_result_meta_data'].get('overall_verdict',[]) == "Total mismatch":
            st.error("Total mismatch")
            st.write(f"Differences: {st.session_state['ga4_result_meta_data']['differences']}")
            st.write(f"Note: {st.session_state['ga4_result_meta_data']['note']}")


        if create_timeline_graph_check and 'ga4_result_table' in st.session_state and 'date' in st.session_state['ga4_result_table'].columns:
            st.session_state['ga4_result_table']['date'] = pd.to_datetime(st.session_state['ga4_result_table']['date'])
            df = st.session_state['ga4_result_table'].set_index('date')
            # 2) Пивот: столбцы = разные pagePath, значения = screenPageViews
            col1, col2, col3 = st.columns(3)
            with col1:
                if len(st.session_state['ga4_result_api']['dimensions']) > 2:
                    dimension_for_chart = st.selectbox("Select the dimension along which the data will be broken.",
                                                       [dim['name'] for dim in st.session_state['ga4_result_api']['dimensions'] if dim['name'] != 'date'])
                else:
                    dimension_for_chart = st.selectbox("Select the dimension along which the data will be broken.",
                                                       [dim['name'] for dim in
                                                        st.session_state['ga4_result_api']['dimensions'] if
                                                        dim['name'] != 'date'])
            with col2:
                if len(st.session_state['ga4_result_api']['metrics']) > 1:
                    metric_for_chart = st.selectbox("Select the metric whose data will be displayed on the graph.",
                                                       [met['name'] for met in
                                                        st.session_state['ga4_result_api']['metrics']])
                else:
                    metric_for_chart = st.selectbox("Select the metric whose data will be displayed on the graph.",
                                                    [met['name'] for met in
                                                     st.session_state['ga4_result_api']['metrics']])


            ts = df.pivot_table(
                index=df.index,
                columns=dimension_for_chart,
                values=metric_for_chart,
                aggfunc='sum'
            )
            ts['Total'] = ts.sum(axis=1)
            if ts['Total'].dtype == 'float64':
                ts['Total'] = ts['Total'].round(2)





            st.write(f"### Тренд «{metric_for_chart}» по «{dimension_for_chart}»")

            df_long = (
                ts
                .reset_index()  # дата из индекса снова — колонка
                .melt(id_vars='date',  # столбец с датой
                      var_name=dimension_for_chart,  # имя новой колонки «значение измерения»
                      value_name=metric_for_chart)  # имя новой колонки «значение метрики»
            )


            # -----------------
            summary = (
                ts
                .sum(axis=0)  # сумма по строкам (дате) → Series
                .reset_index(name='total')  # в DataFrame с колонками [dimension_for_chart, 'total']
            )



            summary.columns = [dimension_for_chart, metric_for_chart + '_total']

            total_row =  summary.loc[summary[dimension_for_chart] == 'Total'].iloc[0].to_dict()

            other_rows = summary[summary[dimension_for_chart] != 'Total'].sort_values(by=[metric_for_chart + '_total'], ascending =False)
            with col3:
                q = st.text_input("Search table:")

                if q:
                    filt = other_rows[other_rows[dimension_for_chart].str.contains(q, case=False, na=False)]
                    other_rows = filt

            # --- 3) Рендерим таблицу со сводкой и включаем выбор строки ---
            gb = GridOptionsBuilder.from_dataframe(other_rows)
            gb.configure_default_column(sortable=True, filter=True)
            gb.configure_selection(selection_mode='multiple', use_checkbox=True)


            grid_opts = gb.build()

            grid_opts['pinnedTopRowData'] = [total_row]

            grid_opts['getRowStyle'] = JsCode(f"""
            function(params) {{
              if (params.node.rowPinned) {{
                return {{ backgroundColor: '#333333' }};
              }}
            }}
            """)


            data_records = other_rows
            col_tab, col_fake= st.columns([1,2])
            with col_tab:
                grid_resp = AgGrid(
                    data_records,
                    fit_columns_on_grid_load=True,
                    gridOptions=grid_opts,
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    allow_unsafe_jscode=True,
                    theme="streamlit"
                )


            selected = grid_resp['selected_rows']
            if type(selected) == pd.DataFrame:
                selected_values = [row for row in selected[dimension_for_chart]]
                selected_values.append('Total')
            # извлекаем выбранное значение измерения
                df_long = df_long[df_long[dimension_for_chart].isin(selected_values) ]
            else:
                df_long = df_long[df_long[dimension_for_chart]=="Total"]



            # -----------------




            # Теперь сам Altair-чарт:
            line = (
                alt.Chart(df_long)
                .mark_line(point=True)  # point=True рисует кружки на точках данных
                .encode(
                    x=alt.X('date:T', title='Дата'),
                    y=alt.Y(f'{metric_for_chart}:Q',
                            title=metric_for_chart,  # например 'activeUsers'
                            scale=alt.Scale(zero=False)  # снимаем принудительный 0, чтобы не было «плоской» линиии
                            ),
                    color=alt.Color(f'{dimension_for_chart}:N', title=dimension_for_chart),  # легенда по измерению
                    strokeDash=alt.condition(
                        alt.datum[dimension_for_chart] == 'Total',
                        alt.value([5, 5]),  # пунктир
                        alt.value([])  # сплошная для остальных
                    ),
                    tooltip=[
                        alt.Tooltip('date:T', title='Дата'),
                        alt.Tooltip(f'{dimension_for_chart}:N', title=dimension_for_chart),
                        alt.Tooltip(f'{metric_for_chart}:Q', title=metric_for_chart, format=',')  # разделитель тысяч
                    ]
                )
                .properties(width=800, height=400)  # подберите свои размеры
                .interactive()  # добавляет zoom & pan
            )
            area = (
                alt.Chart(df_long)
                .mark_area(opacity=0.15)  # 0.15 — 15% непрозрачности, можно варьировать
                .transform_filter(
                    alt.datum[dimension_for_chart] == 'Total'
                )
                .encode(
                    x='date:T',
                    y=alt.Y(f'{metric_for_chart}:Q'),
                    y2=alt.value(0),  # от нуля до линии
                    color=alt.value('#888888')  # общий цвет заливки, можно убрать и Altair возьмёт цвет лин
                )
            )

            # 3) Комбинируем слои
            chart = (area + line).properties(width=800, height=400).interactive()
            with col_fake:
                st.altair_chart(chart, use_container_width=True)










