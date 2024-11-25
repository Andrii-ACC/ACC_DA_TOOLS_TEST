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
from concurrent.futures import ProcessPoolExecutor, as_completed
import gspread
from google.oauth2.service_account import Credentials
from vis_context_tools import promt_chooser,screenshot_by_url, get_text_content_by_url, llm_analysis_of_image, llm_analysis_of_text
from get_clickup_info_by_client import get_ab_test_tickets_info_by_client_name, get_list_of_clients_names

TOOLS_LIST = ["Main page", "Cross-Sales App", "VisContext Analyzer", "Tool number 3"]
DA_NAMES = ["Amar","Djordje","Tarik","other"]
if "CLIENTS_LIST" not in st.session_state.keys() or st.session_state["CLIENTS_LIST"] == None:
    st.session_state['CLIENTS_LIST'] = get_list_of_clients_names()
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
            page_type = st.selectbox(
            "Select the type of page to analyze",
            ["Choose", "Homepage","Product Page","Colection Page","Cart page", "Checkout Page", "CartLayer","Navigation bar","Navigation Layer"]
        )
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
                elif target_audience == "":
                    st.error("Error! The target audience field cannot be empty.")
                elif product_type == "":
                    st.error("Error! The product type field cannot be empty.")
                elif page_type == "Choose":
                    st.error("Error! You must select an analysis page.")
                elif 'url_of_text_content' in locals() and url_of_text_content =='':
                    st.error("Error! You must provide the URL to the page with the tag for analysis.")
                elif 'img_of_content' in locals() and img_of_content ==None:
                    st.error("Error! You must provide a screenshot of the page being analyzed..")
                elif 'url_of_img_content' in locals() and url_of_img_content =='':
                    st.error("You must provide the page URL for visual analysis.")


                prompt_ex_res_list = promt_chooser(analysis_type = analysis_type,
                                                   result_type=result_type,
                                                   company_name = company_name,
                                                   target_audience = target_audience,
                                                   current_date = current_date,
                                                   product_type = product_type,
                                                   page_type = page_type)

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
        import streamlit as st
        import time
        import psutil
        import random
        import os
        import sys
        from PIL import Image, ImageDraw, ImageOps
        from PIL.Image import Resampling
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.service import Service
        from os.path import exists




        # @st.cache_resource
        def get_driver():
            options = webdriver.ChromeOptions()

            options.add_argument('--disable-gpu')
            options.add_argument('--headless')
            options.add_argument(f"--window-size={width}x{height}")

            service = Service()
            driver = webdriver.Chrome(service=service, options=options)

            return webdriver.Chrome(service=service, options=options)


        def get_screenshot(app_url):
            driver = get_driver()
            if app_url.endswith('streamlit.app'):
                driver.get(f"{app_url}/~/+/")
            else:
                driver.get(app_url)

            time.sleep(3)

            # Explicitly wait for an essential element to ensure content is loaded
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            # Get scroll height and width
            # scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
            # scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')

            # Set window size
            # driver.set_window_size(scroll_width, scroll_height)

            # Now, capture the screenshot
            driver.save_screenshot('screenshot.png')


        def add_corners(im, rad):
            circle = Image.new('L', (rad * 2, rad * 2), 0)
            draw = ImageDraw.Draw(circle)
            draw.ellipse((0, 0, rad * 2 - 1, rad * 2 - 1), fill=255)

            alpha = Image.new('L', im.size, 255)
            w, h = im.size

            # Apply rounded corners only to the top
            alpha.paste(circle.crop((0, 0, rad, rad)), (0, 0))
            alpha.paste(circle.crop((rad, 0, rad * 2, rad)), (w - rad, 0))

            im.putalpha(alpha)
            return im


        def generate_app_image():
            bg_random = random.randint(1, 100)
            if bg_random < 10:
                bg_random = '0' + str(bg_random)
            bg_img = Image.open(f'background/background-{bg_random}.jpeg')
            app_img = Image.open('screenshot.png')

            # Create a blank white rectangle
            w, h = app_img.width, app_img.height
            img = Image.new('RGB', (w, h), color='white')

            # Create a drawing object
            draw = ImageDraw.Draw(img)

            # Define the coordinates of the rectangle (left, top, right, bottom)
            rectangle_coordinates = [(0, 0), (w + 50, h + 0)]

            # Draw the white rectangle
            draw.rectangle(rectangle_coordinates, fill='#FFFFFF')
            img = add_corners(img, 24)
            img.save('rect.png')
            ###
            # Resize app image
            image_resize = 0.95
            new_width = int(img.width * image_resize)
            new_height = int(img.height * image_resize)
            resized_app_img = app_img.resize((new_width, new_height))

            # Crop top portion of app_img
            border = (0, 4, 0, 0)  # left, top, right, bottom
            resized_app_img = ImageOps.crop(resized_app_img, border)

            # Add corners
            resized_app_img = add_corners(resized_app_img, 24)

            img.paste(resized_app_img, (int(resized_app_img.width * 0.025), int(resized_app_img.width * 0.035)),
                      resized_app_img)
            img.save('app_rect.png')

            ###
            # Resize app image
            image_resize_2 = 0.9
            new_width_2 = int(bg_img.width * image_resize_2)
            new_height_2 = int(bg_img.height * image_resize_2)
            resized_img = img.resize((new_width_2, new_height_2))

            bg_img.paste(resized_img, (int(bg_img.width * 0.05), int(bg_img.width * 0.06)), resized_img)
            # bg_img.save('final.png')

            if streamlit_logo:
                logo_img = Image.open('streamlit-logo.png').convert('RGBA')
                logo_img.thumbnail([sys.maxsize, logo_width], Resampling.LANCZOS)
                bg_img.paste(logo_img, (logo_horizontal_placement, logo_vertical_placement), logo_img)
                bg_img.save('final.png')

            st.image(bg_img)

            # with Image.open('final.png') as image:
            #    st.image(image)


        # Settings
        with st.sidebar:
            st.header('⚙️ Settings')

            st.subheader('Image Resolution')
            width = st.slider('Width', 426, 1920, 1000)
            height = st.slider('Height', 240, 1080, 540)

            with st.expander('Streamlit logo'):
                streamlit_logo = st.checkbox('Add Streamlit logo', value=True, key='streamlit_logo')
                logo_width = st.slider('Image width', 0, 500, 100, step=10)
                logo_vertical_placement = st.slider('Vertical placement', 0, 1000, 670, step=10)
                logo_horizontal_placement = st.slider('Horizontal placement', 0, 1800, 80, step=10)

            # Getting % usage of virtual_memory ( 3rd field)
            ram_usage = psutil.virtual_memory()[2]
            st.caption(f'RAM used (%): {ram_usage}')

        # Input URL
        with st.form("my_form"):
            app_url = st.text_input('App URL', 'https://langchain-quickstart.streamlit.app').rstrip('/')
            app_name = app_url.replace('https://', '').replace('.streamlit.app', '')

            submitted = st.form_submit_button("Submit")
            if submitted:
                if app_url:
                    get_screenshot(app_url)

        file_exists = exists('screenshot.png')
        if file_exists:
            generate_app_image()

            with open("final.png", "rb") as file:
                btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name=f"{app_name}.png",
                    mime="image/png"
                )
                if btn:
                    os.remove('screenshot.png')
                    os.remove('final.png')









