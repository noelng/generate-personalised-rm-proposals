import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from dotenv import load_dotenv

from multi_doc_rag import MultiDocumentRAG

# Set API keys
load_dotenv()

# Initialize components
search = TavilySearchResults(
    max_results=30,
    time_range="year",
)

llm_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Initialize RAG system
rag_system = MultiDocumentRAG(
    embed_model=embed_model,
    chroma_path="./my_documents_db",
    chunk_size=500,
    chunk_overlap=50
)


def create_hybrid_rm_proposal_analysis(
    company_name: str, 
    web_query: str,
    internal_query: str = None,
    use_vectorstore: bool = True
):
    """
    Generate RM proposal analysis using BOTH web search (Tavily) AND internal documents (RAG)
    
    Args:
        company_name: Name of the company to analyze
        web_query: Query for web search (Tavily)
        internal_query: Query for internal documents (if different from web_query)
        use_vectorstore: Whether to include internal document search
    """
    
    # Step 1: Web Search (Tavily)
    print(f"🔍 [WEB SEARCH] Searching for: {company_name}...")
    web_results = search.invoke({"query": web_query})
    
    # Format web search results
    web_context_parts = []
    for i, result in enumerate(web_results, 1):
        web_context_parts.append(
            f"[Web Source {i}]\n"
            f"Title: {result['title']}\n"
            f"Content: {result['content']}\n"
            f"URL: {result['url']}\n"
            f"Score: {result.get('score', 'N/A')}"
        )
    
    web_context = "\n\n".join(web_context_parts)
    
    # Step 2: Internal Document Search (RAG)
    internal_context = ""
    internal_docs = []
    
    if use_vectorstore:
        try:
            print(f"📚 [INTERNAL DOCS] Searching vectorstore...")
            
            # Use internal_query if provided, otherwise use web_query
            search_query = internal_query if internal_query else web_query
            
            # Get retriever and search
            retriever = rag_system.get_retriever(search_kwargs={"k": 5})
            internal_docs = retriever.invoke(search_query)
            
            # Format internal document results
            internal_context_parts = []
            for i, doc in enumerate(internal_docs, 1):
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                internal_context_parts.append(
                    f"[Internal Doc {i}: {source}]\n"
                    f"Content: {doc.page_content}"
                )
            
            internal_context = "\n\n".join(internal_context_parts)
            print(f"✓ Found {len(internal_docs)} relevant internal documents")
            
        except Exception as e:
            print(f"⚠️ Could not access vectorstore: {e}")
            print("   Continuing with web search only...")
    
    # Step 3: Combine contexts
    if internal_context:
        combined_context = (
            "=== WEB SEARCH RESULTS ===\n\n" + web_context + 
            "\n\n=== INTERNAL COMPANY DOCUMENTS ===\n\n" + internal_context
        )
    else:
        combined_context = web_context
    
    # Step 4: Create enhanced prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior Relationship Manager (RM) analyst for corporate banking in Malaysia.

        Your task: Analyze the provided information from BOTH web sources AND internal company documents 
        to create actionable insights for RM proposals focused on corporate loan opportunities.

        Structure your analysis as follows:

        ## EXECUTIVE SUMMARY
        Brief overview of the company and key findings

        ## COMPANY ANALYSIS
        - Recent developments and activities (from web sources)
        - Historical data and patterns (from internal documents)
        - Financial health indicators
        - Growth trajectory and expansion plans

        ## LOAN OPPORTUNITY ASSESSMENT
        - Specific areas where loan services could be valuable
        - Types of loans that may be relevant (working capital, expansion, refinancing, etc.)
        - Estimated urgency and potential loan size
        - Supporting evidence from both web and internal sources

        ## RM PROPOSAL STRATEGY
        - Recommended approach and talking points
        - Key decision-makers to target
        - Competitive positioning
        - Customization based on historical relationship data

        ## RISK FACTORS
        - Potential concerns or red flags
        - Market or sector risks
        - Historical risk indicators from internal data

        ## NEXT STEPS
        Concrete actions for the RM team

        IMPORTANT: 
        - Cite sources using [Web Source N] for web information and [Internal Doc N] for internal documents
        - Clearly distinguish between public information and internal company data
        - Highlight any contradictions or complementary insights between sources"""),
                ("user", """Company: {company_name}

        Context:
        {context}

        Please provide a comprehensive RM proposal analysis.""")
    ])
    
    # Step 5: Generate analysis
    chain = prompt | llm_model | StrOutputParser()
    
    print("🤖 Generating hybrid analysis...\n")
    analysis = chain.invoke({
        "company_name": company_name,
        "context": combined_context
    })
    
    return analysis, web_results, internal_docs


def save_hybrid_analysis(company_name, analysis, web_results, internal_docs):
    """Save the hybrid analysis to a file"""
    filename = f"{company_name.replace(' ', '_')}_hybrid_analysis.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"HYBRID RM PROPOSAL ANALYSIS: {company_name}\n")
        f.write("="*80 + "\n")
        f.write("(Based on Web Search + Internal Documents)\n")
        f.write("="*80 + "\n\n")
        f.write(analysis)
        f.write("\n\n" + "="*80 + "\n")
        
        # Web sources
        f.write("\n📰 WEB SOURCES:\n")
        for i, source in enumerate(web_results, 1):
            f.write(f"[Web Source {i}] {source['title']}\n")
            f.write(f"               {source['url']}\n\n")
        
        # Internal documents
        if internal_docs:
            f.write("\n📚 INTERNAL DOCUMENTS:\n")
            for i, doc in enumerate(internal_docs, 1):
                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                f.write(f"[Internal Doc {i}] {source}\n")
                f.write(f"                 File type: {doc.metadata.get('file_type', 'N/A')}\n\n")
    
    print(f"✅ Hybrid analysis saved to {filename}")


# Example Usage
if __name__ == "__main__":
    
    # Option 1: First-time setup - Load documents into vectorstore
    # Uncomment this if you need to create/update the vectorstore
    """
    print("="*80)
    print("INITIAL SETUP: Loading Internal Documents")
    print("="*80)
    rag_system.create_vectorstore(directory_path="./my_documents")
    print("\n")
    """
    
    # Option 2: Run hybrid analysis
    company = "Axiata Group Bhd"
    web_query = "Axiata Group Bhd Malaysia news financial performance expansion 2025"
    internal_query = "Axiata customer data credit score financial metrics"
    
    analysis, web_sources, internal_sources = create_hybrid_rm_proposal_analysis(
        company_name=company,
        web_query=web_query,
        internal_query=internal_query,
        use_vectorstore=True  # Set to False to use web search only
    )
    
    # Display results
    print("="*80)
    print(f"HYBRID RM PROPOSAL ANALYSIS: {company}")
    print("="*80)
    print(analysis)
    print("\n" + "="*80)
    
    # Show sources
    print(f"\n📰 Web Sources: {len(web_sources)}")
    print(f"📚 Internal Documents: {len(internal_sources)}")
    
    # Save to file
    save_hybrid_analysis(company, analysis, web_sources, internal_sources)
