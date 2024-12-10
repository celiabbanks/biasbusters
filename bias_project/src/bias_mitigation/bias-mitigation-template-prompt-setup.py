# TEMPLATES SETUP
# RUN BEFORE DATABASE CREATION AND API INTEGRATION

# Templates for implicit bias (educational)
implicit_templates = {
    "bullying": [
        "Let's aim for a positive conversation. Could you rephrase this without hurtful language?",
        "It's important to avoid language that could be interpreted as bullying. Please consider rephrasing."
    ],
    "racial": [
        "This comment might be interpreted as racially biased. How about framing it more inclusively?",
        "Let's try to avoid racially charged language. Could you express your point differently?"
    ],
    "sexist": [
        "Your comment seems to carry gendered bias. Consider using neutral language.",
        "Language free of gender bias helps foster inclusivity. Please rephrase this."
    ],
    "ageism": [
        "Your comment seems to carry ageism bias. Consider using neutral language.",
        "Language free of ageism bias helps foster inclusivity. Please rephrase this."
    ],
    "classist": [
        "Your comment seems to carry classist bias. Consider using neutral language.",
        "Language free of classist bias helps foster inclusivity. Please rephrase this."
    ],
}


# Templates for explicit bias (warnings and guidelines)
explicit_templates = {
    "general": [
        "Your comment violates our community guidelines. Please refrain from using such language.",
        "This comment has been flagged for explicit bias. Continuing this behavior may lead to action."
    ]
}


# Selection Function

import random

def select_template(bias_type, category=None):
    if bias_type == 0:  # Implicit Bias
        if category and category in implicit_templates:
            return random.choice(implicit_templates[category])
        else:
            return "This comment appears biased. Please consider rephrasing."
    elif bias_type == 1:  # Explicit Bias
        return random.choice(explicit_templates["general"])
    else:
        return "This comment could not be processed. Please contact support."


# ChatGPT Prompt Generation

def generate_prompt(body, bias_type, category):
    """
    Generate a prompt for ChatGPT based on bias type and category.
    """
    if bias_type == 0:  # Implicit bias
        template = implicit_templates.get(category, "JenAI says this comment may be biased: '{}'")
    elif bias_type == 1:  # Explicit bias
        template = explicit_templates.get(category, "JenAI says this comment is biased: '{}'")
    else:
        template = "Evaluate this comment for bias: '{}'"

    # Fill the template with the flagged comment
    prompt = template.format(body)
    return prompt

