# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in steramlit for UI/app interface
import streamlit as st

# Import PDF document loaders... There's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import(
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

st.sidebar.markdown('Please add your OpenAI API Key to continue.\n For more information, please check https://platform.openai.com/account/api-keys')
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key"
)
st.markdown('# Good Clinical Practice AI (Q&A)')
st.markdown('Welcome to ChatGPT GCP Edition, a highly advanced artificial intelligence software exclusively trained to help you navigate the complex world of Good Clinical Practice (GCP). Our software uses the state-of-the-art GPT-4 architecture from OpenAI, ensuring you have the most accurate and up-to-date information at your fingertips. \n'
+ 'As a specialized AI, we are equipped to answer a wide range of queries related to GCP, including its principles, guidelines, and the regulations that govern clinical trials across the globe. We can provide assistance with understanding regulatory compliance, ethical standards, data management, informed consent, safety reporting, and much more.\n'
+ 'Whether you\'re a seasoned professional working in the field of clinical research or a newcomer seeking to understand the basics, our AI is designed to facilitate your learning and comprehension of GCP. Please note that while we strive to provide accurate and up-to-date information, our responses should not replace professional legal or medical advice.\n'
+ 'Enjoy exploring the world of GCP with ChatGPT GCP Edition!')

st.markdown('\n')


if len(openai_api_key) != 0 :
    # Set APIKey for OpenAI Service
    # Can sub this out for other LLM providers
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Create instance of OpenAI LLM
    llm = OpenAI(temperature=0.9)

    # Create and load PDF Loader
    loader = PyPDFLoader('ich-guideline-good-clinical-practice-e6r2-step-5_en.pdf')
    # Split pages from pdf
    pages = loader.load_and_split()
    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(pages, collection_name='ich-guideline-good-clinical-practice-e6r2-step-5_en')

    # Create vectorstore info object - metadata repo?
    vectorstore_info = VectorStoreInfo(
        name="Guideline For Good Clinical Practice E6 R2",
        description="Guideline For Good Clinical Practice E6 R2 - Step 5",
        vectorstore=store
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    # Create a text input box for the user
    prompt = st.text_input('Input your GCP question here')

    # If the user hits enter
    if prompt:
        # Then pas sthe prompt to the LLM
        #respose = llm(prompt)
        # Swap out the raw llm for a document agent
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander
        with st.expander('Document Similarity Search'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt)
            # Write out the first
            st.write(search[0][0].page_content)
