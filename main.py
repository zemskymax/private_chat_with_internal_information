import streamlit as st

from langchain_community.llms import Ollama

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


INTERNAL_DATABASE_FOLDER = "internal_db"

MODEL = "llama3"
# MODEL = "phi3"
# MODEL = "gemma"

class CustomLLM:
    def __init__(self):
        raw_prompt = PromptTemplate.from_template(
            """ 
            <s>[INST] You are a technical assistant for question-answering tasks. Search the documents and summarize. 
            If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the answer concise. [/INST]</s>
            [INST]
                Question: {input}
                Context: {context}
                Answer:
            [/INST]
        """
        )

        self.llm = Ollama(model=MODEL, num_ctx=4096, temperature=0.3)

        document_chain = create_stuff_documents_chain(self.llm, raw_prompt)

        # Load vector store
        embedding = FastEmbedEmbeddings()
        vector_store = Chroma(persist_directory=INTERNAL_DATABASE_FOLDER, embedding_function=embedding)

        # Creating chain
        # Only retrieve documents that have a relevance score above a certain threshold
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.2,
            },
    )
        self.chain = create_retrieval_chain(retriever, document_chain)

    #-----------------------#

    def calculate_response(self, prompt):
        response = self.chain.invoke({"input": prompt})

        for doc in response["context"]:
            print("source: " + doc.metadata["source"])

        return response["answer"]

#-----------------------#

def init_stremlit():
    print("-INIT STREAMLIT-")

    # Set page title
    st.session_state.app_name = "⚡️ Private Chat With Internal Knowledge"
    st.title(f"{st.session_state.app_name}")

    # Add BOT (LLM) greeting (if needed)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi There, how can I help you?"}]

    # Write message history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

#-----------------------#

def handle_user_input():
    print("-HANDLE USER INPUT-")

    custom_llm = CustomLLM()

    if prompt := st.chat_input("Enter your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user", avatar="🧑‍💻").write(prompt)

        # Clear the previous message
        st.session_state["full_message"] = ""

        print(prompt)

        # Write the generated response
        result = custom_llm.calculate_response(prompt)

        print(result)

        st.session_state["full_message"] = result
        st.chat_message("assistant", avatar="🤖").write(st.session_state["full_message"]) # use write_stream instead of write

        # Update message history
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})

#-----------------------#

def main():
    print("-MAIN-")

    init_stremlit()
    handle_user_input()

    return 0

if __name__ == '__main__':
    main()
