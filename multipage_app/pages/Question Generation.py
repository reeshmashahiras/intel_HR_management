import streamlit as st


st.title("Question Generation")

st.write("You have entered", st.session_state["my_input"])
# streamlit_app.py
import streamlit as st
from transformers import pipeline
import spacy

# Install necessary packages
# !pip install transformers spacy
# !python -m spacy download en_core_web_sm

# Load spacy and question generation model
nlp = spacy.load("en_core_web_sm")
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def extract_answers(context):
    doc = nlp(context)
    return [ent.text for ent in doc.ents]

def generate_question_answer_pairs(context, answers):
    qa_pairs = []
    for answer in answers:
        input_text = f"answer: {answer} context: {context}"
        generated_questions = question_generator(input_text, max_length=64)
        qa_pairs.append({"question": generated_questions[0]["generated_text"], "answer": answer})
    return qa_pairs

def main():
    # Initialize session state
    if "my_input" not in st.session_state:
        st.session_state.my_input = ""

    # Streamlit app title and description
    st.title("QA Generation Test")
    st.write("Enter a context or upload a document to see the generated questions and answers.")

    # Option to upload a document
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

    # Option to enter text manually
    text_input = st.text_area("Enter the text manually:")

    # Determine the context based on user input
    if uploaded_file is not None:
        # Read the content of the uploaded file with explicit encoding
        file_contents = uploaded_file.read().decode("utf-8", "ignore")
        st.session_state.my_input = file_contents
    elif text_input:
        # Use manually entered text as the context
        st.session_state.my_input = text_input

    # Process when the user submits the form
    if st.button("Generate QA Pairs"):
        answers = extract_answers(st.session_state.my_input)
        qa_pairs = generate_question_answer_pairs(st.session_state.my_input, answers)

        # Display QA pairs in a table
        st.table(qa_pairs)

if __name__ == "__main__":
    main()
