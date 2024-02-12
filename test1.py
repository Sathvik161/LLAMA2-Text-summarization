import streamlit as st
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
import PyPDF2
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from docx import Document
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message
import tempfile


# Streamlit app title and description
st.title("Text Summarization and Q&A")
st.write("Upload multiple documents (PDF, Word, or Txt) or enter text below for summarization and Q&A.")

# Sidebar for selecting writing style and adjusting chunk size
st.sidebar.title("Modes")
writing_style = st.sidebar.radio("Choose Writing Style", ["Creative", "Normal", "Academic"],index=1)

# Map CHUNK_SIZE values to labels
chunk_size_labels = {"Small": 4096, "Medium": 2370, "Large": 512}

# Add a select_slider for selecting the chunk size label
selected_chunk_size_label = st.sidebar.select_slider(
    "Select Summary Size",
    options=list(chunk_size_labels.keys()),
    value="Medium"
)

# Get the corresponding CHUNK_SIZE value
CHUNK_SIZE = chunk_size_labels[selected_chunk_size_label]


# Create the LlamaCpp instance
callback = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="PROVIDE-WITH-YOUR-LLAMA2-MODEL-PATH",
    temperature=0.5,
    n_gpu_layers=50,
    n_batch=4096,
    n_ctx=4096,
    max_tokens=4096,
    top_p=1,
    callback_manager=callback,
    verbose=True
)

# Define prompt template based on selected writing style
prompt_templates = {
    "Creative":  """
    Be creative and provide a summary for the following text using your unique writing style.
    {text}
    CREATIVE SUMMARY:
    """,
    "Normal": """
    Write a concise summary of the following text delimited by triple backticks in a single line.
    {text}
    CONCISE SUMMARY:
    """,
    "Academic":  """
    Summarize the text in an academic style using proper language and structure.
    {text}
    ACADEMIC SUMMARY:
    """
}

template = prompt_templates.get(writing_style, "")

prompt = PromptTemplate(template=template, input_variables=["text"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function to extract text from different document types
def extract_text_from_file(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        return extract_text_from_docx(uploaded_file)
    elif file_extension == "txt":
        return extract_text_from_txt(uploaded_file)
    else:
        # Handle other document types as needed
        st.warning(f"Unsupported file type: {file_extension}")
        return ""

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

def extract_text_from_docx(docx_file):
    text = ""
    doc = Document(docx_file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        text = file.read()
    return text

def calculate_cosine_similarity(reference_text, generated_summary):
    vectorizer = CountVectorizer().fit_transform([reference_text, generated_summary])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity(vectors[0].reshape(1, -1), vectors[1].reshape(1, -1))
    similarity_percentage = similarity[0][0] * 100  # Convert to percentage
    return similarity_percentage

# Function to display original document and its summary
def display_summary(file_name, original_text, summary_text):
    st.subheader(f"--- {file_name} ---")
    
    # Display original document
    st.write("Original Document:")
    st.markdown(original_text, unsafe_allow_html=True)

    # Display summary
    st.write("Summary:")
    st.write(summary_text)

# Function to summarize input text
def summarize_input_text(direct_text):
    text_name = "Text_Input"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50, separators="\n\n")
    generated_summary = "" 
    for chunk in text_splitter.split_text(direct_text):
        result = llm_chain(chunk)
        summary_text = result.get("text", "\n\n")
        generated_summary += summary_text
        st.write(summary_text)
    
    # Calculate and display cosine similarity for direct text as a percentage
    similarity_percentage = calculate_cosine_similarity(direct_text, generated_summary)
    st.write(f"Accuracy for Direct Text: {similarity_percentage:.2f}%")


# Function to summarize input document
def summarize_input_document(uploaded_files):
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_text = extract_text_from_file(uploaded_file)
        # Process and print the summary for each chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)
        num_chunks = len(list(text_splitter.split_text(file_text)))
        progress_bar = st.progress(0)

        # Process and print the summary for each chunk
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=50)

        # Display original document alongside its summary
        st.subheader(f"--- {file_name} ---")
        
        # Display the original document as an expandable component
        with st.expander("View Original Document"):
            st.markdown(file_text, unsafe_allow_html=True)

        # Display summary
        st.write("Summary:")
        generated_summary = "" 
        for i, chunk in enumerate(text_splitter.split_text(file_text), start=1):
            result = llm_chain(chunk)
            summary_text = result.get("text", "\n\n")
            generated_summary += summary_text
            st.write(summary_text)
            progress_bar.progress(i / num_chunks)

        # Calculate and display cosine similarity as a percentage
        similarity_percentage = calculate_cosine_similarity(file_text, generated_summary)
        st.write(f"Accuracy: {similarity_percentage:.2f}%")


        # Add a separator line between files
        st.write("---")


doc_type = option_menu(
    menu_title = "",
    options = ["Text","PDF", "Word", "Txt"],
    icons = None,
    #default_index = 0,
    orientation = "horizontal"
)

# Main section based on document type
if doc_type == "Text":
    st.title("Enter text for summarization")
    direct_text = st.text_area("Enter Text for Summarization")
    if st.button("SUMMARIZE"):
        summarize_input_text(direct_text)

elif doc_type in ["PDF", "Word", "Txt"]:
    st.title(f"Upload {doc_type.lower()} document for summarization")
    uploaded_files = st.file_uploader(f"Upload {doc_type.lower()} file(s)", type=[doc_type.lower()], accept_multiple_files=True)
    if st.button("SUMMARIZE"):
        summarize_input_document(uploaded_files)

# End of the code

 #Q&A

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

def create_conversational_chain(vector_store):
    # Create llm
    # Constants
    MODEL_PATH = "PROVIDE-WITH-YOUR-LLAMA2-MODEL-PATH"
    N_GPU_LAYERS = 50
    N_BATCH = 1024
    N_CTX = 2048
    MAX_TOKENS = 100
    CHUNK_SIZE = 200  # Adjust the chunk size as needed

    # Create the LlamaCpp instance
    callback = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.5,
    n_gpu_layers=N_GPU_LAYERS,
    n_batch=N_BATCH,
    n_ctx=N_CTX,
    max_tokens=MAX_TOKENS,
    top_p=1,
    callback_manager=callback,
    verbose=True
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)
    return chain

def main():
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Q&A bot")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                          model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)

if __name__ == "__main__":
    main()  
