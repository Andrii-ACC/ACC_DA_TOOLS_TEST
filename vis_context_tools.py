import base64
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from crewai import Crew, Agent, Task,Process
from langchain_openai import ChatOpenAI
import streamlit as st

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager











def promt_chooser (analysis_type,result_type,company_name,target_audience,current_date,product_type,page_type,text_content = "Some Text Content"):
    prompt = None
    expected_output = None
    if analysis_type == "Text Content Analysis":

        if result_type == "Strengths and Weaknesses Analysis":
            prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name} online store, which sells {product_type}, aimed at {target_audience}.

Focus on the following aspects:
- Clarity, relevance, and appeal of product descriptions and other textual content
- Effectiveness of calls-to-action (CTAs) in encouraging conversions for this target audience
- Use of psychological triggers relevant to the audience (e.g., urgency, social proof, trust-building)
- Overall readability and logical flow of the content

Identify the strengths and weaknesses of the text content and suggest improvements that would increase user engagement and conversions.

TEXT CONTENT ON THE {page_type} OF {company_name}:\n
{text_content}
"""
            expected_output = f"""Strengths and Weaknesses Analysis:

- Introduction: Briefly summarize the effectiveness of the page’s written content in engaging the target audience and guiding them toward conversion.
- Strengths: List effective text elements, such as clarity in product descriptions, impactful calls-to-action (CTAs), and well-used psychological triggers. Each strength should be clear and concise, highlighting the specific benefit for the audience.
- Weaknesses: Identify problematic text elements, including vague descriptions, ineffective CTAs, or missing motivational triggers. Each weakness should detail the specific issue and its potential negative impact on user engagement or conversions.
- Suggestions for Improvement: Recommend specific text improvements based on weaknesses, such as adjusting language to be more action-oriented or adding urgency in CTAs.
- Conclusion: Recap the key findings and emphasize the expected impact of improving text clarity, persuasiveness, and alignment with user motivations."""
        elif result_type == "Hypothesis Formation for A/B Testing":
            prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name} (selling {product_type}) for {target_audience} , and generate 3-5 hypotheses for A/B testing that could increase conversions. Each hypothesis should include:

- The specific text element to test (e.g., CTA wording, product description phrasing)
- A rationale explaining why this change may improve user engagement or conversions for this audience
- The expected outcome of the test (e.g., increased click-through rate, reduced bounce rate)

TEXT CONTENT ON THE {page_type} OF {company_name}:\n
{text_content}
"""
            expected_output = f"""Hypothesis Formation for A/B Testing:

Introduction: Summarize the primary text issues identified that may benefit from testing.\n
Hypotheses: Present a series of hypotheses in the following format:
- If: Clearly state the proposed text change (e.g., rephrasing CTA, adding social proof language).
- Then: Outline the anticipated improvement in user behavior or conversion.
- Because: Briefly explain the reasoning, supported by user behavior principles (e.g., urgency increases action, trust language increases confidence).\n
Conclusion: Highlight the potential outcomes of testing these hypotheses for optimizing conversions."""


    if analysis_type == "Full Screenshot Analysis":
        if result_type == "Strengths and Weaknesses Analysis":
            prompt = f"""Today is {current_date}. Conduct a visual analysis of the {page_type} for {company_name}, who sales {product_type}, designed for {target_audience}.

Focus on:
- Visual hierarchy: How well does the layout guide users’ attention to essential elements?
- Color scheme: Is the color palette engaging and aligned with the product and target audience?
- White space usage: Is the design clear and uncluttered, promoting easy navigation?
- CTA visibility and placement: Are calls-to-action prominent and intuitively located?

Provide an overview of the strengths and weaknesses in the visual design and layout, along with suggestions for improving user experience and engagement.
"""
            expected_output = f"""Strengths and Weaknesses Analysis:

- Introduction: Provide an overview of how the page's overall design supports or detracts from user engagement and conversion goals.
- Strengths: List visually effective elements such as intuitive layout, engaging color scheme, or high-impact images that create a strong visual hierarchy. Each strength should clarify its role in enhancing the user experience.
- Weaknesses: Identify design aspects that might hinder engagement, such as poor contrast, lack of CTA visibility, or overwhelming clutter. Each weakness should specify how it could lead to user drop-off or reduced interaction.
- Suggestions for Improvement: Provide actionable design recommendations to address weaknesses, such as adjusting color contrast for better visibility or improving white space for a cleaner look.
- Conclusion: Summarize the expected benefits of implementing these design improvements for increased user engagement."""

        elif result_type == "Hypothesis Formation for A/B Testing":
            prompt = f"""Today is {current_date}. Analyze the visual elements on the {page_type} for {company_name} online store , which sales {product_type}, designed for {target_audience} , and suggest 3-5 hypotheses for A/B testing.

Each hypothesis should include:
- The design element to test (e.g., color of CTA buttons, layout adjustments)
- Rationale on how the change could positively impact user behavior (e.g., increased visibility of key information)
- Expected outcome (e.g., improved click-through rates, longer time on page)

Example: "Testing a bolder color for the CTA button may increase click-throughs by making the action more visible and urgent."
"""
            expected_output = f"""Hypothesis Formation for A/B Testing:

Introduction: Summarize key design issues that may benefit from testing.\n
Hypotheses: List hypotheses for A/B testing in the following structure:
- If: Describe the proposed design modification (e.g., repositioning CTA, adjusting color scheme).
- Then: Specify the expected behavioral change (e.g., more clicks on CTA, longer session duration).
- Because: Offer a short explanation for the hypothesis, grounded in design psychology (e.g., better CTA visibility prompts faster action).\n
Conclusion: Emphasize the potential for improved conversions based on successful testing outcomes."""


    if analysis_type=="Section-Specific Screenshot Analysis":
        if result_type == "Strengths and Weaknesses Analysis":
            prompt = f"""Today is {current_date}. Analyze the provided {page_type} section on the website of the {company_name} online store, selling {product_type}, aimed at {target_audience}. Evaluate each element of the section (e.g. image, product catalog, footer, buttons, etc.) separately, as well as all the relationships between the elements.

For each element of the section, focus on:
- Ease of understanding and accessibility of key information
- Elements that capture user attention and those that may go unnoticed
- Effectiveness of psychological triggers for this audience (e.g., urgency, social proof)

Identify the strengths and weaknesses of each element of the section and suggest improvements to enhance user engagement and conversions.
"""
            expected_output = f"""Strengths and Weaknesses Analysis:

- Introduction: Briefly introduce the analysis of specific elements in the section and their roles in supporting user flow and conversion goals.
- Strengths: For each element, list effective subelements. Each strength should outline how it contributes positively to the user experience.
- Weaknesses: For each element, identify areas needing improvement (e.g., missing CTAs, confusing navigation). Each weakness should detail its impact on user interaction or engagement.
- Suggestions for Improvement: Recommend targeted changes for each element oe entire section based on weaknesses, such as adding a prominent CTA  or simplifying product card information.
- Conclusion: Recap how optimizing specific elements could enhance the page’s usability and conversion rates."""
        elif result_type == "Hypothesis Formation for A/B Testing":
            prompt = f"""Today is {current_date}. Analyze the provided {page_type} section on the website of the {company_name}, selling {product_type}, aimed at {target_audience} and generate 3-5 hypotheses for A/B testing to improve user engagement.

Each hypothesis should include:
- The specific section or element to test (e.g.,  image, CTA in product catalog etc)
- Reasoning for why this change may benefit the target audience’s interaction and decision-making
- Expected impact (e.g., increased interaction with key sections, higher conversion rates)"""
            expected_output = f"""Hypothesis Formation for A/B Testing:

Introduction: Provide a summary of section-specific issues that could be tested to optimize user engagement.\n
Hypotheses: List hypotheses in the following structure, specifying the section for each:
- If: Describe the proposed change within a specific element or section (e.g., adding social proof to the product grid, placing CTA in hero section).
- Then: Define the expected user response (e.g., increased product clicks, reduced bounce rate).
- Because: Briefly explain the reasoning, connecting the change to user behavior (e.g., social proof increases trust in products).\n
Conclusion: Emphasize the benefits of testing these section-specific changes to enhance page performance and conversions."""

    return [prompt,expected_output]
def screenshot_by_url(website: str):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1200')
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager(driver_version='120.0.6099.224').install()),
                              options=options)



    driver.get(website)
    time.sleep(1)
    width = driver.execute_script(
        "return Math.max( document.body.scrollWidth, document.body.offsetWidth, document.documentElement.clientWidth, document.documentElement.scrollWidth, document.documentElement.offsetWidth );")
    height = driver.execute_script(
        "return Math.max( document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight );")
    driver.set_window_size(width, height)
    time.sleep(1)

    return driver.find_element(By.TAG_NAME, 'body').screenshot_as_base64  # avoids scrollbar

def get_text_content_by_url (url : str):

    options = Options()
    options.add_argument('--headless=new')
    driver = webdriver.Chrome( options=options)
    driver.get(url)

    # Закрываем драйвер
    html_content = driver.page_source
    soup = BeautifulSoup(html_content, "html.parser")
    for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
        script_or_style.decompose()
    text_elements = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "span", "div"]):
        if tag.get_text(strip=True): # Проверяем, что внутри есть текст
            clean_tag = f"<{tag.name}>{tag.get_text(strip=True)}</{tag.name}>"
            text_elements.append(str(clean_tag))




    driver.quit()
    return  "\n".join(text_elements)
def llm_analysis_of_image(img , prompt):
    client = OpenAI()
    response = client.chat.completions.create(
        temperature=0.25,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a Conversion Rate Optimization (CRO) Specialist."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""{prompt}""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content
def llm_analysis_of_text(context, prompt :str, expected_output):
    prompt = prompt.replace("Some Text Content",context)
    visual_agent = Agent(
        role='Conversion Rate Optimization (CRO) Specialist',
        backstory='You is an advanced in UI/UX optimization and conversion rate analysis, with deep knowledge of psychological triggers, user behavior, and best practices for web design. You has experience analyzing e-commerce platforms and understands how specific design elements impact user decisions.',
        goal='Your task is to assess online store webpages, extract and analyze text and visual elements, identify strong and weak design aspects, and provide actionable insights to optimize the user experience and increase conversions. This includes generating hypotheses for A/B testing, evaluating psychological triggers, and suggesting improvements.',
        llm=ChatOpenAI(model_name="gpt-4o", temperature=0.5),
        tools = []
    )
    analyzing_text_content_and_structure = Task(
        description=f"""{prompt}""",
        expected_output=f"""{expected_output}"""
    )
    crew = Crew(
        agents=[visual_agent],
        tasks=[analyzing_text_content_and_structure],
        verbose=True,
        process=Process.hierarchical,
        manager_llm=ChatOpenAI(model="gpt-4o"),
        max_rpm=30,
        memory=True,
        cache=True
    )
    result = crew.kickoff()

    return result.raw

