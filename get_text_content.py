from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from crewai_tools import SeleniumScrapingTool
from crewai import Crew, Agent, Task,Process
import os
from langchain_openai import ChatOpenAI


OpenAIGPT = ChatOpenAI(model_name="gpt-4o", temperature=0.5)
url = "https://acceleratedagency.com/"
css = 'section[class]'
def count_of_element_by_url (url):

    options = Options()
    options.add_argument('--headless=new')
    driver = webdriver.Chrome( options=options)
    driver.get(url)

    # Находим все элементы, соответствующие селектору section[class]
    elements = driver.find_elements(By.CSS_SELECTOR, css_selector)

    # Выводим количество найденных элементов
    number_of_elements = len(elements)

    # Закрываем драйвер
    driver.quit()
    '''context_list =[]
    for section in range(number_of_elements):
        tool = SeleniumScrapingTool(
            website_url=url,
            cookie={'name': 'user', 'value': 'John Doe'},
            wait_time=1,
            css_element=f"{css_selector}:nth-child({section + 1})"
        )
        context_list.append(f'Content of Section {section + 1}\n________________\n{tool.run()}\n________________')'''
    context_list = SeleniumScrapingTool(
            website_url=url,
            cookie={'name': 'user', 'value': 'John Doe'},
            wait_time=5,
            css_element=f"document"
        )
    return context_list.run()



visual_agent = Agent(
    role = 'Digital Experience Optimizer and UX/UI Consultant',
    backstory = 'You is an advanced in UI/UX optimization and conversion rate analysis, with deep knowledge of psychological triggers, user behavior, and best practices for web design. You has experience analyzing e-commerce platforms and understands how specific design elements impact user decisions.',
    goal = 'Your task is to assess online store webpages, extract and analyze text and visual elements, identify strong and weak design aspects, and provide actionable insights to optimize the user experience and increase conversions. This includes generating hypotheses for A/B testing, evaluating psychological triggers, and suggesting improvements.',
    llm = OpenAIGPT,

)

analyzing_text_content_and_structure = Task(
    description = f"""Analyze the text content of this e-commerce webpage. Focus on:

- Clarity and appeal of product descriptions
- Effectiveness of calls-to-action (CTA) in encouraging purchases
- Use of psychological triggers (e.g., urgency, social proof, trust-building elements)
- Logical flow and readability of the text

Identify strengths and weaknesses in the text content and suggest improvements to enhance user engagement and drive conversions.

Text content of this e-commerce webpage:\n
{count_of_element_by_url(url,css)}""",
    expected_output = """Strength: Description of the strengths of the text content in terms of conversion rate optimization
Weakness: Description of the weaknesses of the text content in terms of conversion rate optimization
Improve: Tips for improving the structure and context of the page in order to increase the conversion rate"""
)
crew = Crew(
            agents=[visual_agent],
            tasks=[analyzing_text_content_and_structure],
            verbose=True,
            process=Process.hierarchical,
            manager_llm=ChatOpenAI(model="gpt-4o"),
            max_rpm = 30,
            memory=True,
            cache = True
        )
res = crew.kickoff()
print(res)