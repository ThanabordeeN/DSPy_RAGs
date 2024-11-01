from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import SQLiteVec
from langchain_text_splitters import CharacterTextSplitter


class SQLiteVecManager:
    def __init__(self,table_name="markdown",db_file="vec.db", chunk_size=1000, chunk_overlap=200, model="nomic-embed-text"):
        self.db_file = db_file
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = model
        self.embedding_function = OllamaEmbeddings(model=model)
        self.connection = SQLiteVec.create_connection(db_file=db_file)
        self.db = SQLiteVec(
            table=table_name, embedding=self.embedding_function, connection=self.connection
        )

    def query_text(self, query):
        data = self.db.similarity_search(query)
        return data

    def add_text(self, file_path):
        loader = DirectoryLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in docs]

        self.db.add_texts(texts)
        return True
    
    
# if __name__ == "__main__":
#     db = SQLiteVecManager()
#     db.add_text("data/")
    