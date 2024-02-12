
# Multi-Document-Summarization-using-LLAMA2 🦙

In numerous domains, a vast amount of information is accessible, posing challenges for individuals to sift through and distill pertinent insights. Parsing through lengthy documents or numerous articles is a time-intensive task. Extracting relevant data from a pool of documents demands substantial manual effort and can be quite challenging. Hence, our project, Multiple Document Summarization Using Llama 2, proposes an initiative to address these issues. We aim to summarize extensive documents or data sets efficiently, providing users with concise and relevant summaries. This approach facilitates the extraction of pertinent insights, comprehension of lengthy documents, and eliminates the need for excessive manual effort.

## Problem Statement

Unlocking the Power of LLAMA2 for Multi Document Summarization and Q&A Chatbot Integration.

## Solution

We employ Llama2 as the primary Large Language Model for our Multiple Document Summarization task. Developed by Meta AI, Llama2 is an open-source model released in 2023, proficient in various natural language processing (NLP) tasks, such as text generation, text summarization, question answering, code generation, and translation.

For Multiple Document Summarization, Llama2 extracts text from the documents and utilizes an Attention Mechanism to generate the summary. This mechanism functions by enabling the model to comprehend the context and relationships between words, akin to how the human brain prioritizes important information when reading a sentence. Essentially, the attention mechanism identifies the most crucial sentences within a text by assigning weights to each word based on its significance. Words with higher weights are deemed more important, guiding the model's focus during summary generation.

## Tech Stack
- **StreamLit** : Front-end framework 
- **LLAMA2** : Large Language Model as a service 
- **LangChain** : Open-source framework designed to facilitate decentralized translation and language services.
- **Scikit-Learn**


## Workflow

![image](https://github.com/katakampranav/Multi-Document-Summarization-using-LLAMA2/assets/133202118/cd83d1e1-1d77-4a7e-b19a-9c1708a66dc6)


## Multi Document Summarization

## User Interface:
### 1. Creation of Interface:

The user interface is developed using Streamlit, a Python library for building interactive web applications.\
Components such as titles, descriptions, and sidebars are utilized to structure the interface.

### 2. Selection Options:

Users are provided with options to select their preferred writing style (creative, normal, academic) via a radio button.\
They can also adjust the summary size using a select slider, with options for small, medium, and large summaries.

### 3. Input Methods:

- The interface offers two methods for input :
  - #### Upload Documents:
       Users can upload text documents (PDF, Word, or Txt) using file upload widgets.

  - #### Direct Input: 
       Alternatively, users can enter text directly into a text area provided for summarization.

![image](https://github.com/katakampranav/Multi-Document-Summarization-using-LLAMA2/assets/133202118/7c1a8027-f84f-4a74-b695-d45267491d4e)

## Text Summarization Logic:
### 1. Text Extraction:

Functions are defined to extract text from various types of documents (PDF, Word, Txt).\
This involves reading and processing the contents of uploaded documents or direct text input.

### 2. Cosine Similarity Calculation:

After generating the summary, cosine similarity is computed between the original text and the generated summary.\
Cosine similarity is a metric used to measure the similarity between two vectors, often used in text analysis to compare documents or sentences.

### 3. Display Functions:

Functions are provided to display the original document alongside its summary.\
This allows users to compare the generated summary with the original text.

### 4. Summarization Process:

#### The logic for summarizing input text and documents involves:
- #### Chunking: 
     The text/documents are split into smaller chunks for processing.

- #### Model Inference:
     Each chunk is fed into the llama2 model to generate a summary.

- #### Display of Summaries: 
     The generated summaries are displayed alongside the original documents.

### 5. Iterative Process:

Summarization is performed iteratively for each chunk of text, ensuring comprehensive coverage of the input documents.\
The llama2 model is utilized for generating summaries, with each chunk processed individually to maintain coherence and accuracy.


![image](https://github.com/katakampranav/Multi-Document-Summarization-using-LLAMA2/assets/133202118/3202083e-8f09-4b64-a246-6adb6aca8a4c)


## QnA Chatbot

### 1. Initialization:
- The initialize_session_state() function sets up the session state for the conversation history and generated responses.
- It initializes empty lists to store the chat history, past user inputs, and generated responses.

### 2. Conversation Chat:
- The conversation_chat() function handles user queries by feeding them to the conversational retrieval chain.
- It takes the user query, the conversation chain, and the chat history as inputs.
- It appends the user query and the generated response to the chat history.

### 3. Display Chat History:
- The display_chat_history() function manages the display of the chat history and user interactions.
- It provides a text input field for users to ask questions.
- Upon submitting a question, it triggers the conversation with the conversational chain and displays the response.
- It visualizes the conversation history, showing both user queries and model responses.

### 4. Creating Conversational Chain:
- The create_conversational_chain() function sets up the conversational retrieval chain.
- It initializes the llama2 model and necessary parameters for conversational retrieval.
- The chain is configured with a memory component to maintain chat history.

![image](https://github.com/katakampranav/Multi-Document-Summarization-using-LLAMA2/assets/133202118/ce57c72b-f399-41c3-aca6-adfd8e4ef482)

## Application Significance:

There is a growing need for efficient text summarization, particularly in scenarios where users deal with large volumes of textual data. This need is emphasized in fields such as:
### 1. News Aggregation:
•	Scenario: News outlets receive vast amounts of information daily.\
•	Usage: Summarizing news articles can help users quickly grasp the key points without reading lengthy articles.
### 2. Legal Documents:
•	Scenario: Law firms deal with extensive legal documents.\
•	Usage: Summarization can extract crucial details, making it more efficient for lawyers to review and understand case materials.
### 3. Academic Research:
•	Scenario: Researchers analyse numerous papers for literature reviews.\
•	Usage: Summarization aids in extracting essential findings and methodologies from multiple research papers.
### 4. Business Reports:
•	Scenario: Companies generate extensive reports for decision-making.\
•	Usage: Summarizing business reports can provide executives with concise insights and key metrics.
### 5. E-learning Platforms:
•	Scenario: Students need to process a large volume of educational content.\
•	Usage: Summarization can help in condensing study materials, making it easier for students to review and understand key concepts.
### 6. Medical Records:
•	Scenario: Healthcare professionals deal with extensive patient records.\
•	Usage: Summarization can assist in extracting critical information from patient histories for quick and accurate decision-making.


## Author

- [@Sathvik161](https://github.com/Sathvik161)
- **Repositories :**
https://github.com/Sathvik161/LLAMA2-Text-summarization

## Feedback 

If you have any feedback, please reach out me at sathvik.vittapu@gmail.com
