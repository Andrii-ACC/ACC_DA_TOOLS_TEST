
def promt_chooser (analysis_type,result_type,company_name,target_audience,current_date,product_type,page_type,text_content):
    prompt = None
    expected_output = None
    if analysis_type == "Text Content Analysis":

        if result_type == "Strengths and Weaknesses Analysis Prompt":
            prompt = f"""Today is {current_date}. Analyze the text content on the {page_type} of {company_name}, aimed at {target_audience}, which sells {product_type}.

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

Introduction: Briefly summarize the effectiveness of the page’s written content in engaging the target audience and guiding them toward conversion.
Strengths: List effective text elements, such as clarity in product descriptions, impactful calls-to-action (CTAs), and well-used psychological triggers. Each strength should be clear and concise, highlighting the specific benefit for the audience.
Weaknesses: Identify problematic text elements, including vague descriptions, ineffective CTAs, or missing motivational triggers. Each weakness should detail the specific issue and its potential negative impact on user engagement or conversions.
Suggestions for Improvement: Recommend specific text improvements based on weaknesses, such as adjusting language to be more action-oriented or adding urgency in CTAs.
Conclusion: Recap the key findings and emphasize the expected impact of improving text clarity, persuasiveness, and alignment with user motivations."""
        elif result_type == "Hypothesis Formation for A/B Testing Prompt":
            prompt = f"""Today is {current_date}. Based on an analysis of the text content on the {page_type} of {company_name} for {target_audience} (selling {product_type}), generate 3-5 hypotheses for A/B testing that could increase conversions. Each hypothesis should include:

- The specific text element to test (e.g., CTA wording, product description phrasing)
- A rationale explaining why this change may improve user engagement or conversions for this audience
- The expected outcome of the test (e.g., increased click-through rate, reduced bounce rate)"
"""
            expected_output = f"""Hypothesis Formation for A/B Testing:

Introduction: Summarize the primary text issues identified that may benefit from testing.
Hypotheses: Present a series of hypotheses in the following format:
-If: Clearly state the proposed text change (e.g., rephrasing CTA, adding social proof language).
-Then: Outline the anticipated improvement in user behavior or conversion.
-Because: Briefly explain the reasoning, supported by user behavior principles (e.g., urgency increases action, trust language increases confidence).
Conclusion: Highlight the potential outcomes of testing these hypotheses for optimizing conversions."""
    if analysis_type == "Full Screenshot Analysis":
        if result_type == "Strengths and Weaknesses Analysis Prompt":
            prompt = f"""Today is {current_date}. Conduct a visual analysis of the {page_type} for {company_name}, designed for {target_audience}, who purchase {product_type}.

Focus on:
- Visual hierarchy: How well does the layout guide users’ attention to essential elements?
- Color scheme: Is the color palette engaging and aligned with the product and target audience?
- White space usage: Is the design clear and uncluttered, promoting easy navigation?
- CTA visibility and placement: Are calls-to-action prominent and intuitively located?

Provide an overview of the strengths and weaknesses in the visual design and layout, along with suggestions for improving user experience and engagement.
"""
            expected_output = f"""Strengths and Weaknesses Analysis:

Introduction: Provide an overview of how the page's overall design supports or detracts from user engagement and conversion goals.
Strengths: List visually effective elements such as intuitive layout, engaging color scheme, or high-impact images that create a strong visual hierarchy. Each strength should clarify its role in enhancing the user experience.
Weaknesses: Identify design aspects that might hinder engagement, such as poor contrast, lack of CTA visibility, or overwhelming clutter. Each weakness should specify how it could lead to user drop-off or reduced interaction.
Suggestions for Improvement: Provide actionable design recommendations to address weaknesses, such as adjusting color contrast for better visibility or improving white space for a cleaner look.
Conclusion: Summarize the expected benefits of implementing these design improvements for increased user engagement."""
        elif result_type == "Hypothesis Formation for A/B Testing Prompt":
            prompt = f"""Today is {current_date}. Based on an analysis of the visual elements on the {page_type} for {company_name}, designed for {target_audience} interested in {product_type}, suggest 3-5 hypotheses for A/B testing.

Each hypothesis should include:
- The design element to test (e.g., color of CTA buttons, layout adjustments)
- Rationale on how the change could positively impact user behavior (e.g., increased visibility of key information)
- Expected outcome (e.g., improved click-through rates, longer time on page)

Example: "Testing a bolder color for the CTA button may increase click-throughs by making the action more visible and urgent."
"""
            expected_output = f"""Hypothesis Formation for A/B Testing:

Introduction: Summarize key design issues that may benefit from testing.
Hypotheses: List hypotheses for A/B testing in the following structure:
-If: Describe the proposed design modification (e.g., repositioning CTA, adjusting color scheme).
-Then: Specify the expected behavioral change (e.g., more clicks on CTA, longer session duration).
-Because: Offer a short explanation for the hypothesis, grounded in design psychology (e.g., better CTA visibility prompts faster action).
Conclusion: Emphasize the potential for improved conversions based on successful testing outcomes."""
    if analysis_type=="Section-Specific Screenshot Analysis":
        if result_type == "Strengths and Weaknesses Analysis Prompt":
            prompt = f"""Today is {current_date}. Analyze the individual sections of the {page_type} on {company_name}’s website, targeted at {target_audience} interested in {product_type}. Evaluate each section (e.g., hero image, product catalog, footer) separately.

For each section, focus on:
- Ease of understanding and accessibility of key information
- Elements that capture user attention and those that may go unnoticed
- Effectiveness of psychological triggers for this audience (e.g., urgency, social proof)

Identify the strengths and weaknesses of each section and suggest improvements to enhance user engagement and conversions.
"""
            expected_output = f"""Strengths and Weaknesses Analysis:

Introduction: Briefly introduce the analysis of specific sections on the page (e.g., hero section, product grid, footer) and their roles in supporting user flow and conversion goals.
Strengths: For each section, list effective elements (e.g., a visually appealing hero section, clear product grid). Each strength should outline how it contributes positively to the user experience.
Weaknesses: For each section, identify areas needing improvement (e.g., missing CTAs, confusing navigation). Each weakness should detail its impact on user interaction or engagement.
Suggestions for Improvement: Recommend targeted changes for each section based on weaknesses, such as adding a prominent CTA in the hero section or simplifying product card information.
Conclusion: Recap how optimizing specific sections could enhance the page’s usability and conversion rates."""
        elif result_type == "Hypothesis Formation for A/B Testing Prompt":
            prompt = f"""Today is {current_date}. Analyze the individual sections of the {page_type} on {company_name}’s website, targeted at {target_audience} interested in {product_type}. Evaluate each section (e.g., hero image, product catalog, footer) separately.

For each section, focus on:
- Ease of understanding and accessibility of key information
- Elements that capture user attention and those that may go unnoticed
- Effectiveness of psychological triggers for this audience (e.g., urgency, social proof)

Identify the strengths and weaknesses of each section and suggest improvements to enhance user engagement and conversions.
"""
            expected_output = f"""Hypothesis Formation for A/B Testing:

Introduction: Provide a summary of section-specific issues that could be tested to optimize user engagement.
Hypotheses: List hypotheses in the following structure, specifying the section for each:
-If: Describe the proposed change within a specific section (e.g., adding social proof to the product grid, placing CTA in hero section).
-Then: Define the expected user response (e.g., increased product clicks, reduced bounce rate).
-Because: Briefly explain the reasoning, connecting the change to user behavior (e.g., social proof increases trust in products).
Conclusion: Emphasize the benefits of testing these section-specific changes to enhance page performance and conversions."""

    return [prompt,expected_output]