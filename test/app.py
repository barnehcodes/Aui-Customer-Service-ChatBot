#!/usr/bin/env python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
import json

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("./fine-tuned-model")
tokenizer = T5Tokenizer.from_pretrained("./fine-tuned-model")

# Function to generate responses
def generate_response(user_input):
    try:
        # Add context to the input
        prompt = "Answer the following question about university admissions: "
        input_text = prompt + user_input

        # Tokenize the input
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        # Generate the output
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=5,
            early_stopping=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Fallback for short or generic responses
        if len(response.split()) < 5:
            response = "I'm sorry, I couldn't find a detailed answer. Please visit our admissions page or contact the admissions office for more information."

        return response
    except Exception as e:
        return "An error occurred. Please try again later."

# Gradio chat interface function
def respond(message, history):
    response = generate_response(message)
    return response

# Create a Gradio ChatInterface
interface = gr.ChatInterface(
    fn=respond,
    title="University Customer Support Chatbot",
    description="Ask me anything about admissions, courses, or campus life!",
    examples=[
        "What are the admission requirements?",
        "Do I need to submit SAT scores for admission?",
        "What is the minimum GPA required for admission?"
    ]
)

# Launch the interface
interface.launch()