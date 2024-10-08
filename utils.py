def generate_rephrased_questions(question):
    """Generate multiple rephrased versions of a given question using predefined templates."""
    rephrased_templates = [
        "Can you explain about {}?",
        "What can you tell me about {}?",
        "Please provide details on {}.",
        "I'd like to know more about {}.",
        "Could you give an overview of {}?"
    ]
    rephrased_questions = [template.format(question) for template in rephrased_templates]
    return rephrased_questions

def create_detailed_context(context, question):
    """Combine the original context with additional context and rephrased questions to create a detailed context for querying."""
    rephrased_questions = generate_rephrased_questions(question)
    additional_context = f"Additional context about {question}."
    full_context = context + "\n" + additional_context + "\n" + "\n".join(rephrased_questions)
    return full_context
