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











def promt_chooser (analysis_type,result_type,company_name,target_audience,current_date,product_type,page_type, industry_type, promt_chooser_radio, text_content = "Some Text Content"):
    prompt = None
    expected_output = None
    if promt_chooser_radio == "Andrii":
        if analysis_type == "Text Content Analysis":

            if result_type == "Strengths and Weaknesses Analysis":
                prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name} online store in the {industry_type}, which sells {product_type}, aimed at {target_audience}.
    
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
                prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name} online store in the {industry_type}(selling {product_type}) for {target_audience} , and generate 3-5 hypotheses for A/B testing that could increase conversions. Each hypothesis should include:
    
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
                prompt = f"""Today is {current_date}. Conduct a visual analysis of the {page_type} for {company_name} online store in the {industry_type}, who sales {product_type}, designed for {target_audience}.
    
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
                prompt = f"""Today is {current_date}. Analyze the visual elements on the {page_type} for {company_name} online store in the {industry_type}, which sales {product_type}, designed for {target_audience} , and suggest 3-5 hypotheses for A/B testing.
    
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
                prompt = f"""Today is {current_date}. Analyze the provided {page_type} section on the website of the {company_name} online store in the {industry_type}, selling {product_type}, aimed at {target_audience}. Evaluate each element of the section (e.g. image, product catalog, footer, buttons, etc.) separately, as well as all the relationships between the elements.
    
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
                prompt = f"""Today is {current_date}. Analyze the provided {page_type} section on the website of the {company_name} online store in the {industry_type}, selling {product_type}, aimed at {target_audience} and generate 3-5 hypotheses for A/B testing to improve user engagement.
    
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
    elif promt_chooser_radio == "Denis":

        if analysis_type == "Text Content Analysis":

            if result_type == "Strengths and Weaknesses Analysis":
                prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name} online store in the {industry_type}, which sells {product_type}, aimed at {target_audience}.
    
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
    
    Introduction
      Briefly summarize the effectiveness of the {industry_type} - specific {product_type} page’s written content in engaging the target audience and guiding them toward conversion. Mention any industry standards or benchmarks that are relevant.
    
    Strengths
      List the effective text elements tailored to the {industry_type}, such as:
      - Clarity in {product_type}-specific descriptions that address {industry_type} user needs.
      - Impactful calls-to-action (CTAs) that resonate with {industry_type} user motivations.
      - Well-used psychological triggers pertinent to {industry_type} customers, such as social proof, authority, scarcity, and exclusivity.
      Each strength should be clear and concise, highlighting the specific benefit for the {industry_type} audience.
    
    Weaknesses
      Identify problematic text elements, specific to the {industry_type} & {product_type}, including:
      - Vague descriptions that fail to address key {product_type} and  {industry_type} pain points or benefits.
      - Ineffective CTAs that do not compel the target audience to act.
      - Missing motivational triggers crucial for conversions.
      Each weakness should detail the specific issue and its potential negative impact on user engagement, trust, or conversions within the {industry_type}.
    
    Suggestions for Improvement
      Recommend specific, actionable text improvements based on identified weaknesses, such as:
      - Adjusting language to be more action-oriented and appealing to users.
      - Incorporating urgency and exclusivity in CTAs to drive immediate action within the {industry_type}.
      - Enhancing product descriptions to better address customer pain points and needs.
    
    Conclusion
    Recap the key findings, emphasizing the expected impact of improving text clarity, persuasiveness, and alignment with user motivations. Highlight potential improvements in conversion rates and overall user engagement for the {product_type} page within the {industry_type}.
    """
            elif result_type == "Hypothesis Formation for A/B Testing":
                prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name} online store in the {industry_type}(selling {product_type}) for {target_audience} , and generate 3-5 hypotheses for A/B testing that could increase conversions. Each hypothesis should include:
    
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
                prompt = f"""Today is {current_date}. Conduct a visual analysis of the {page_type} for {company_name} online store in the {industry_type}, who sales {product_type}, designed for {target_audience}.
    Analyze the visual design and layout of the e-commerce store's full-page screenshot focusing on the following aspects. Provide insights grounded in sales psychology principles, and suggest actionable steps to improve user experience and conversion rates.
    
    Visual Hierarchy
    - Evaluate how effectively the layout guides users' attention to essential elements such as product images, descriptions, and CTAs.
    - Identify any distractions or elements that may detract from the primary focus areas.
    
    Color Scheme
    - Assess the engagement level and alignment of the color palette with the product and target audience.
    - Determine if the colors evoke the desired emotional response and support brand perception.
    
    White Space Usage
    - Examine the clarity and navigational ease provided by the use of white space in the design.
    - Identify areas that appear cluttered or overwhelming.
     
    CTA Visibility and Placemen
    - Analyze the prominence, visibility, and intuitive placement of calls-to-action.
    - Evaluate if the CTAs stand out and are easily accessible during the user journey.
    
    Overview of Strengths and Weaknesses
    - **Strengths:**
    - Highlight effective design elements that successfully capture user attention and facilitate navigation.
    - Discuss areas where color scheme, visual hierarchy, or CTA placement are particularly effective in driving user actions.
    
    - **Weaknesses:**
    -- Identify design flaws that may hinder user engagement or conversion, such as poorly placed CTAs or distracting elements.
    -- Discuss any inconsistencies in color scheme or cluttered areas that detract from the user experience.
    
    - **Suggestions for Improvement:**
    -- Provide specific, actionable steps to refine the visual hierarchy, color scheme, white space usage, and CTA placement.
    -- Emphasize how each suggested change can enhance user experience, induce the desired emotional response, and ultimately improve conversion rates.
    
    Conclusion
    - Summarize the key insights from the analysis and restate the most critical improvements needed.
    - Highlight the expected impact of these changes on user engagement and conversion rates, reinforcing the importance of a user-centric and psychologically-informed design strategy.
    
    """
                expected_output = f"""Strengths and Weaknesses Analysis:
    Introduction
    - Provide an overview of how the page's overall design either supports or detracts from user engagement and conversion goals. Discuss how design elements align with sales psychology principles, such as guiding attention, evoking emotional responses, and facilitating decision-making.
    
    Strengths
    - List visually effective elements and explain their role in enhancing the user experience. Focus on:
    -- **Intuitive Layout:** How well the layout guides users through the page and highlights essential elements like products and CTAs.
    -- **Engaging Color Scheme:** How the color palette aligns with the brand and evokes the desired emotional response from the target audience.
    -- **High-Impact Images:** The effectiveness of images in creating a strong visual hierarchy and resonating with users' emotions and motivations.
    - Each strength should include a brief explanation of its psychological impact on user engagement and conversion.
    
    Weaknesses
    - Identify design aspects that might hinder user engagement and conversions. Focus on:
    -- **Poor Contrast:** Issues with color contrast that may reduce text or CTA visibility and readability.
    -- **Lack of CTA Visibility:** Instances where calls-to-action are not prominent or intuitively placed, leading to potential user drop-off.
    -- **Overwhelming Clutter:** Areas where excessive design elements create confusion or hinder easy navigation.
    - Each weakness should specify how it could negatively impact user behavior and conversion rates.
    
    Suggestions for Improvement
    - Provide actionable design recommendations to address identified weaknesses. Focus on:
    -- **Adjusting Color Contrast:** Enhancing visibility for better readability and CTA prominence.
    -- **Improving CTA Placement:** Making calls-to-action more prominent by using contrasting colors, larger buttons, and intuitive positioning.
    -- **Increasing White Space:** Simplifying the layout to create a cleaner look and improve user focus and navigation.
    - Include reasoning based on sales psychology principles, such as the importance of clear visual hierarchy and the role of cognitive ease in user decision-making.
    
    Conclusion
     Summarize the expected benefits of implementing these design improvements, such as increased user engagement, higher trust, and better conversion rates. Emphasize the importance of aligning design elements with sales psychology to create a more compelling and effective user experience.
    """


            elif result_type == "Hypothesis Formation for A/B Testing":
                prompt = f"""Today is {current_date}. Analyze the visual elements on the {page_type} for {company_name} online store in the {industry_type}, which sales {product_type}, designed for {target_audience} , and suggest 3-5 hypotheses for A/B testing.
    
    Each hypothesis should include:
    - The design element to test (e.g., color of CTA buttons, layout adjustments)
    - Rationale on how the change could positively impact user behavior (e.g., increased visibility of key information)
    - Expected outcome (e.g., improved click-through rates, longer time on page)
    
    Example: "Testing a bolder color for the CTA button may increase click-throughs by making the action more visible and urgent."
    """
                expected_output = f"""Hypothesis Formation for A/B Testing:
    Introduction
    - Summarize key design issues identified through the full screenshot analysis that may benefit from A/B testing. Highlight how addressing these issues can align with sales psychology principles to enhance user engagement and conversion rates.
    
    Hypotheses
    - List hypotheses for A/B testing using the following structure:
    
    -- IF Describe the proposed design modification (e.g., repositioning CTA, adjusting color scheme).
    
    -- THEN Specify the expected behavioral change (e.g., more clicks on CTA, longer session duration).
    
    -- BECAUSE Offer a short explanation for the hypothesis, grounded in design and sales psychology (e.g., better CTA visibility prompts faster action).
    
    
    
    Conclusion
      Emphasize the potential benefits of testing these design modifications. Discuss how successful testing outcomes can lead to improved conversions, enhanced user experience, and better alignment with user behavior principles.
    """


        if analysis_type=="Section-Specific Screenshot Analysis":
            if result_type == "Strengths and Weaknesses Analysis":
                prompt = f"""Today is {current_date}. Analyze the provided {page_type} section on the website of the {company_name} online store in the {industry_type}, selling {product_type}, aimed at {target_audience}. Evaluate each element of the section (e.g. image, product catalog, footer, buttons, etc.) separately, as well as all the relationships between the elements.
    
    Analyze each element of the specified section, focusing on its impact on user engagement and conversion rates. Consider the following aspects to provide a comprehensive evaluation:
    
    Ease of Understanding and Accessibility
    - Assess how easily users can comprehend and access key information.
    - Determine whether the text is clear, concise, and free of jargon.
    - Evaluate the layout and design elements that contribute to or hinder understanding and accessibility.
    
    Elements that Capture User Attention
    - Identify which elements successfully capture and retain user attention.
    - Highlight any elements that may go unnoticed or overlooked.
    - Consider visual hierarchy, design prominence, and the use of color and typography.
    
    Effectiveness of Psychological Triggers
    - Examine the presence and effectiveness of psychological triggers aimed at influencing user behavior.
    - Consider triggers such as urgency (e.g., limited-time offers), social proof (e.g., reviews, testimonials), and authority (e.g., expert endorsements).
    - Assess how well these triggers are integrated into the overall design and messaging.
    
    Identify the strengths and weaknesses of each element of the section and suggest improvements to enhance user engagement and conversions.
    """
                expected_output = f"""Strengths and Weaknesses Analysis:
    
    Introduction
    - Briefly introduce the analysis of specific elements within the section. Discuss how these elements support user flow and contribute to conversion goals, emphasizing the importance of optimized design in enhancing user engagement and driving conversions.
    
    Strengths
    - For each element, list and describe the subelements that effectively enhance the user experience. Focus on:
    -- How each strength contributes to ease of understanding and accessibility.
    -- Elements that successfully capture user attention.
    -- Effective use of psychological triggers that positively influence user behavior.
    
    Weaknesses
    - For each element, identify and describe areas that need improvement. Focus on:
    -- Specific issues such as missing calls-to-action (CTAs), confusing navigation, or poor contrast.
    -- The impact of these weaknesses on user interaction, engagement, and conversion rates.
    -- Any elements that might go unnoticed or fail to capture user attention effectively.
    
    Suggestions for Improvement
    - Recommend targeted changes for each element or the entire section based on identified weaknesses. Provide actionable and specific recommendations, such as:
    -- Adding prominent CTAs to increase user action.
    -- Simplifying product card information to improve clarity.
    -- Enhancing color contrast for better visibility and user focus.
    -- Improving navigation to make it more intuitive and user-friendly.
    
    Conclusion
    - Recap the key findings and suggestions. Highlight how optimizing specific elements can enhance the page’s usability, increase user engagement, and improve conversion rates. Emphasize the expected benefits of implementing the suggested changes, supported by sales psychology principles.
    """
            elif result_type == "Hypothesis Formation for A/B Testing":
                prompt = f"""Today is {current_date}. Analyze the provided {page_type} section on the website of the {company_name} online store in the {industry_type}, selling {product_type}, aimed at {target_audience} and generate 3-5 hypotheses for A/B testing to improve user engagement.
    
    Each hypothesis should include the following components to ensure a well-rounded evaluation and clear action plan:
    
    Specific Section or Element to Test
    - Clearly identify the section or element to be tested (e.g., product images, CTA in product catalog, navigation menu).
    
    Reasoning
    - Explain why this change may benefit the target audience's interaction and decision-making.
    - Include insights from user behavior principles and sales psychology, such as how the change addresses current pain points or leverages psychological triggers.
    
    Expected Impact
    - Outline the anticipated improvement in user behavior or conversion metrics.
    - Be specific about the expected outcomes, such as increased interaction with key sections, higher conversion rates, or improved user satisfaction.
    """
                expected_output = f"""Hypothesis Formation for A/B Testing:
    Introduction
    - Provide a summary of section-specific issues that could be tested to optimize user engagement. Highlight how these issues impact user behavior and conversion goals.
    
    Hypotheses
    - List hypotheses using the following structure, specifying the section for each test:
    
    -- IF Describe the proposed change within a specific element or section (e.g., adding social proof to the product grid, placing CTA in hero section).
       
    -- THEN Define the expected user response (e.g., increased product clicks, reduced bounce rate).
       
    -- BECAUSE Briefly explain the reasoning, connecting the change to user behavior and sales psychology principles (e.g., social proof increases trust in products).
    
    Conclusion
    - Emphasize the benefits of testing these section-specific changes. Discuss how these optimizations can enhance page performance and increase conversions, supported by principles of user behavior and sales psychology.
    """

        return [prompt,expected_output]
    else:
        return ["Prompt not found", "Expected output not found"]
def screenshot_by_url(website: str):
    chromium_path = '/usr/bin/chromium'
    chrome_driver_path = '/usr/bin/chromedriver'
    options = Options()
    options.binary_location = chromium_path
    options.add_argument('--headless')  # Run in headless mode.
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')  # Bypass OS security model.
    options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems.
    options.add_argument('--window-size=1920,1200')

    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)




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
    chromium_path = '/usr/bin/chromium'
    chrome_driver_path = '/usr/bin/chromedriver'
    options = Options()
    options.binary_location = chromium_path
    options.add_argument('--headless')  # Run in headless mode.
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')  # Bypass OS security model.
    options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource problems.
    options.add_argument('--window-size=1920,1200')

    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
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

