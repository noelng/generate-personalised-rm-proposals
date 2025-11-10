from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import your models
from multi_doc_rag import MultiDocumentRAG

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  

### embedding model
embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize components
rag_system = MultiDocumentRAG(
    embed_model=embed_model,
    chroma_path="C:/Users/noeln/OneDrive/Desktop/Agentic RAG/generate-personalised-rm-proposals/my_documents_db",
    chunk_size=500,
    chunk_overlap=50
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)

# Step 1: Create vector store (run once)
print("=" * 80)
print("STEP 1: Loading Documents")
print("=" * 80)

# Option A: Load from directory
rag_system.create_vectorstore(directory_path="C:/Users/noeln/OneDrive/Desktop/Agentic RAG/generate-personalised-rm-proposals/my_documents")

# Option B: Load specific files
# rag_system.create_vectorstore(document_paths=[
#     "./documents/annual_report.pdf",
#     "./documents/financial_data.xlsx",
#     "./documents/meeting_notes.docx",
#     "./documents/customer_feedback.csv"
# ])


"""
Below are the steps for setting up the RAG chain and querying it. Uncomment to run.
"""

# Step 2: Create RAG chain
# print("\n" + "=" * 80)
# print("STEP 2: Setting up RAG Chain")
# print("=" * 80)

# retriever = rag_system.get_retriever(search_kwargs={"k": 5})

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """You are a helpful assistant that answers questions based on the provided context.
    
#     Use the following pieces of context to answer the question. If you don't know the answer 
#     based on the context, say so - don't make up information.

#     Always cite which document(s) you're referring to when answering.

#     Context:
#     {context}"""),
#         ("user", "{question}")
#     ])

# def format_docs(docs):
#     """Format retrieved documents for the prompt."""
#     formatted = []
#     for i, doc in enumerate(docs, 1):
#         source = os.path.basename(doc.metadata.get('source', 'Unknown'))
#         formatted.append(f"[Document {i}: {source}]\n{doc.page_content}")
#     return "\n\n".join(formatted)

# # Create the RAG chain
# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt_template
#     | llm
#     | StrOutputParser()
# )

# Step 3: Query the system
# print("\n" + "=" * 80)
# print("STEP 3: Querying the System")
# print("=" * 80)

# questions = [
#     "What are the key financial highlights from the documents?",
#     "Summarize the main findings",
#     "What are the customer feedback trends?",
#     "List out all documents and their key points mentioned."
# ]

# for question in questions:
#     print(f"\n❓ Question: {question}")
#     print("-" * 80)
    
#     # Get answer
#     answer = rag_chain.invoke(question)
#     print(f"💡 Answer:\n{answer}\n")