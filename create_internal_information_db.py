import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


INTERNAL_INFORMATION_FOLDER = "internal_information"
INTERNAL_INFORMATION_EXTENSION = "pdf"
INTERNAL_DATABASE_FOLDER = "internal_db"


def process_internal_information():
    print("-PROCESS INTERNAL INFORMATION-")

    # Find all the relevant internal files
    internal_files = []
    for filename in os.listdir(INTERNAL_INFORMATION_FOLDER):
        f = os.path.join(INTERNAL_INFORMATION_FOLDER, filename)
        # checking if it is a file
        if os.path.isfile(f) and f.endswith(INTERNAL_INFORMATION_EXTENSION):
            print(f)
            internal_files.append(f)

    # Parse the internal files to retrieve all the information
    file_counter = 0
    total_internal_documents = []
    for internal_file in internal_files:
        file_counter += 1
        loader = PDFPlumberLoader(internal_file)
        docs = loader.load()
        total_internal_documents = total_internal_documents + docs
        print(f"File #{file_counter} contains {len(docs)} pages (documents)")

    print(f"Total internal pages (documents) = {len(total_internal_documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64, length_function=len, is_separator_regex=False)
    total_internal_chunks = text_splitter.split_documents(total_internal_documents)
    print(f"Total internal chunks = {len(total_internal_chunks)}")

    # Save the internal information in a local database
    embedding = FastEmbedEmbeddings()
    vector_store = Chroma.from_documents(documents=total_internal_chunks, embedding=embedding, persist_directory=INTERNAL_DATABASE_FOLDER)
    vector_store.persist()

#-----------------------#

def main() -> int:
    print("-MAIN-")

    process_internal_information()

    return 0

if __name__ == '__main__':
    main()
