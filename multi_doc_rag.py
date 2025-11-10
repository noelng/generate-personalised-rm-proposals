import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# ✅ Text splitter (moved from langchain to langchain_text_splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ✅ Vector store (modularized)
from langchain_chroma import Chroma
# ✅ Document loaders (moved to langchain_community)
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
)
# ✅ Document schema (moved to langchain_core)
from langchain_core.documents import Document

# Import your embedding model
# from src.models.model import embed_model

load_dotenv()


class MultiDocumentRAG:
    """RAG system that handles multiple document types."""
    
    SUPPORTED_EXTENSIONS = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,
        '.xlsx': UnstructuredExcelLoader,
        '.xls': UnstructuredExcelLoader,
        '.csv': CSVLoader,
        '.txt': TextLoader,
        '.md': TextLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
    }
    
    def __init__(
        self, 
        embed_model,
        chroma_path: str = "./chroma_langchain_db",
        collection_name: str = "multi-doc-rag",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """Initialize the RAG system."""
        self.embed_model = embed_model
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document based on its file extension."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )
        
        loader_class = self.SUPPORTED_EXTENSIONS[extension]
        
        try:
            print(f"Loading {file_path.name}...")
            
            # Special handling for CSV files
            if extension == '.csv':
                loader = loader_class(file_path=str(file_path))
            else:
                loader = loader_class(str(file_path))
            
            documents = loader.load()
            
            # Add source metadata
            for doc in documents:
                doc.metadata['source'] = str(file_path)
                doc.metadata['file_type'] = extension
            
            print(f"✓ Loaded {len(documents)} document(s) from {file_path.name}")
            return documents
            
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory."""
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_documents = []
        
        # Find all supported files
        for extension in self.SUPPORTED_EXTENSIONS.keys():
            files = list(directory.glob(f"**/*{extension}"))
            
            for file_path in files:
                docs = self.load_document(str(file_path))
                all_documents.extend(docs)
        
        print(f"\n📚 Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def create_vectorstore(
        self, 
        documents: Optional[List[Document]] = None,
        document_paths: Optional[List[str]] = None,
        directory_path: Optional[str] = None
    ):
        """Create or update vector store with documents."""
        
        # Determine document source
        if documents is None:
            documents = []
            
            if document_paths:
                for path in document_paths:
                    docs = self.load_document(path)
                    documents.extend(docs)
            
            elif directory_path:
                documents = self.load_directory(directory_path)
            
            else:
                raise ValueError(
                    "Must provide either 'documents', 'document_paths', or 'directory_path'"
                )
        
        if not documents:
            raise ValueError("No documents to process")
        
        # Split documents
        print("\n✂️  Splitting documents into chunks...")
        doc_splits = self.text_splitter.split_documents(documents)
        print(f"✓ Created {len(doc_splits)} chunks")
        
        # Check if vectorstore exists
        if os.path.exists(self.chroma_path):
            print(f"\n📦 Loading existing vector store from {self.chroma_path}...")
            vectorstore = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embed_model,
                collection_name=self.collection_name,
            )
            
            # Add new documents
            print("➕ Adding new documents to existing vector store...")
            vectorstore.add_documents(doc_splits)
        else:
            print(f"\n🆕 Creating new vector store at {self.chroma_path}...")
            vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=self.embed_model,
                persist_directory=self.chroma_path,
            )
        
        print("✅ Vector store ready!")
        return vectorstore
    
    def load_vectorstore(self):
        """Load existing vector store."""
        if not os.path.exists(self.chroma_path):
            raise FileNotFoundError(
                f"Vector store not found at {self.chroma_path}. "
                "Please create it first using create_vectorstore()"
            )
        
        print(f"📦 Loading vector store from {self.chroma_path}...")
        vectorstore = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embed_model,
            collection_name=self.collection_name,
        )
        
        return vectorstore
    
    def get_retriever(self, search_kwargs: dict = None):
        """Get retriever from vector store."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}  # Return top 5 results
        
        vectorstore = self.load_vectorstore()
        return vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def query(self, question: str, k: int = 5) -> List[Document]:
        """Query the vector store."""
        vectorstore = self.load_vectorstore()
        results = vectorstore.similarity_search(question, k=k)
        
        print(f"\n🔍 Found {len(results)} relevant chunks:")
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"  {i}. {Path(source).name}")
        
        return results
    
    def delete_vectorstore(self):
        """Delete the vector store."""
        import shutil
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
            print(f"🗑️  Deleted vector store at {self.chroma_path}")
        else:
            print("No vector store to delete")


# Example usage
# if __name__ == "__main__":
#     from src.models.model import embed_model  # Your embedding model
    
#     # Initialize RAG system
#     rag = MultiDocumentRAG(
#         embed_model=embed_model,
#         chroma_path="./my_document_store",
#         collection_name="my-documents",
#         chunk_size=500,
#         chunk_overlap=50
#     )
    
    # Example 1: Load documents from a directory
    # rag.create_vectorstore(directory_path="./documents")
    
    # Example 2: Load specific files
    # rag.create_vectorstore(document_paths=[
    #     "./data/report.pdf",
    #     "./data/analysis.docx",
    #     "./data/data.xlsx"
    # ])
    
    # Example 3: Query the vector store
    # results = rag.query("What are the key findings?", k=3)
    
    # Example 4: Get retriever for RAG chain
    # retriever = rag.get_retriever(search_kwargs={"k": 5})
