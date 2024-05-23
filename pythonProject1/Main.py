import getpass
import pickle
import PyPDF2
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_community.vectorstores import FAISS
import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

API_O = st.sidebar.text_input("API-KEY")


# if "generated" not in st.session_state:
#     st.session_state["generated"] = []
# if "past" not in st.session_state:
#     st.session_state["past"] = []
# if "input" not in st.session_state:
#     st.session_state["input"] = ""
# if "stored_session" not in st.session_state:
#     st.session_state["stored_session"] = []

langchain.verbose = False

# load env varibales
load_dotenv()

# def new_chat():
#     """
#     Clears session state and starts a new chat.
#     """
#     save = []
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         save.append("User:" + st.session_state["past"][i])
#         save.append("Bot:" + st.session_state["generated"][i])
#     st.session_state["stored_session"].append(save)
#     st.session_state["generated"] = []
#     st.session_state["past"] = []
#     st.session_state["input"] = ""
#     st.session_state.entity_memory.entity_store = {}

with st.sidebar.expander("🛠️ ", expanded=False):


    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])


# process text from pdf
def process_text(text):
    # split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_text(text)

    # convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(texts, embeddings)

    return knowledge_base


def main():

    st.title("Chat with my PDF")
    os.environ["OPENAI_API_KEY"] = API_O
    #os.environ["OPENAI_API_KEY"] = getpass.getpass()

    pdf = st.file_uploader("Upload your PDF File", type="pdf")

    if pdf is not None:

        pdf_reader = PyPDF2.PdfReader(pdf)

        # store the pdf text in a var
        text = ""

        for page in pdf_reader.pages:

            text += page.extract_text()
        pdf.close()


            # convert the chunks of text into embeddings to form a knowledge base
        embeddings = OpenAIEmbeddings()

        knowledge_base = process_text(text)

        query = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your AI assistant here! Ask me anything ...",
                            label_visibility='hidden')

        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query is not None:
            docs = knowledge_base.similarity_search(query)


            # llm = OpenAI(openai_api_key='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')
            # #
            # chain = load_qa_chain(llm, chain_type="stuff")
            # #
            # with get_openai_callback() as cost:
            #     response = chain.run(input_documents=docs, question=query)
            #     print(cost)

            # st.session_state.past.append(query)
            # st.session_state.generated.append(docs)
            st.write(docs)


if __name__ == "__main__":
    main()