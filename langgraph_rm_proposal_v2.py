import os
from typing import TypedDict, List, Dict
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import json

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


# Define the state
class RMProposalState(TypedDict):
    """State for the RM proposal generation workflow"""
    company_name: str
    web_query: str
    use_vectorstore: bool
    
    # Results from each step
    web_results: list
    web_context: str
    
    # Loan product recommendations from web analysis
    suggested_loan_products: List[str]  # e.g., ["Working Capital Loan", "Trade Finance"]
    
    # Product info sheets from vectorstore
    product_info_docs: list
    product_info_context: str
    
    # Combined contexts
    combined_context: str
    
    # Final output
    analysis: str
    error: str


# Node 1: Web Search
def web_search_node(state: RMProposalState) -> RMProposalState:
    """Perform web search using Tavily"""
    print(f"🔍 [WEB SEARCH] Searching for: {state['company_name']}...")
    
    try:
        web_results = search.invoke({"query": state['web_query']})
        
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
        
        print(f"✓ Found {len(web_results)} web sources")
        
        return {
            **state,
            "web_results": web_results,
            "web_context": web_context,
            "error": ""
        }
    
    except Exception as e:
        print(f"⚠️ Web search error: {e}")
        return {
            **state,
            "web_results": [],
            "web_context": "",
            "error": f"Web search failed: {str(e)}"
        }


# Node 2: Identify Loan Products
def identify_loan_products_node(state: RMProposalState) -> RMProposalState:
    """Analyze web results and identify suitable loan products"""
    
    print("💡 [LOAN ANALYSIS] Identifying suitable loan products...")
    
    try:
        # Create a prompt to analyze company needs and suggest loan products
        loan_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a corporate banking expert specializing in loan product matching.

            Based on the company's recent activities, financial situation, and business developments from web sources, identify which loan products would be most suitable.

            Common loan products include:
            - Working Capital Loan / Revolving Credit
            - Term Loan / Business Expansion Loan
            - Trade Finance / Letter of Credit
            - Project Finance
            - Equipment Financing / Asset-Based Lending
            - Bridge Loan / Short-term Financing
            - Refinancing Facility
            - SME Loan / Enterprise Financing
            - Property Development Loan
            - Export Credit / Import Financing

            Analyze the company's:
            1. Business activities (expansion, acquisitions, operations)
            2. Financial needs (cash flow, capital requirements)
            3. Industry sector and typical financing needs
            4. Growth stage and development plans

            Output ONLY a JSON array of 2-4 most relevant loan product names, for example:
            ["Working Capital Loan", "Trade Finance", "Business Expansion Loan"]

            No explanations, just the JSON array."""),
                        ("user", """Company: {company_name}

            Web Search Results:
            {web_context}

            Identify the most suitable loan products for this company.""")
        ])
        
        chain = loan_analysis_prompt | llm_model | StrOutputParser()
        response = chain.invoke({
            "company_name": state['company_name'],
            "web_context": state['web_context'][:8000]  # Limit context size
        })
        
        # Parse JSON response
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response.replace("```json", "").replace("```", "").strip()
            
            suggested_products = json.loads(response)
            
            if not isinstance(suggested_products, list):
                suggested_products = []
        
        except json.JSONDecodeError:
            print(f"⚠️ Could not parse loan products, using defaults")
            suggested_products = ["Working Capital Loan", "Business Expansion Loan"]
        
        print(f"✓ Identified {len(suggested_products)} loan products:")
        for product in suggested_products:
            print(f"   - {product}")
        
        return {
            **state,
            "suggested_loan_products": suggested_products
        }
    
    except Exception as e:
        print(f"⚠️ Loan analysis error: {e}")
        return {
            **state,
            "suggested_loan_products": ["Working Capital Loan", "Business Expansion Loan"]
        }


# Node 3: Retrieve Product Info Sheets
def retrieve_product_info_node(state: RMProposalState) -> RMProposalState:
    """Search vectorstore for loan product information sheets"""
    
    if not state['use_vectorstore']:
        print("⏭️  [PRODUCT INFO] Skipping vectorstore search (disabled)")
        return {
            **state,
            "product_info_docs": [],
            "product_info_context": ""
        }
    
    print(f"📋 [PRODUCT INFO] Searching for product information sheets...")
    
    try:
        product_info_docs = []
        
        # Search for each suggested loan product
        for product_name in state['suggested_loan_products']:
            # Create search query for product info sheet
            query = f"{product_name} product information sheet eligibility criteria requirements"
            print(f"   Searching: {product_name}...")
            
            retriever = rag_system.get_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(query)
            product_info_docs.extend(docs)
        
        # Remove duplicates based on content
        seen_content = set()
        unique_docs = []
        for doc in product_info_docs:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        product_info_docs = unique_docs
        
        # Format product info results
        product_info_parts = []
        for i, doc in enumerate(product_info_docs, 1):
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            product_info_parts.append(
                f"[Product Info {i}: {source}]\n"
                f"Content: {doc.page_content}"
            )
        
        product_info_context = "\n\n".join(product_info_parts)
        print(f"✓ Found {len(product_info_docs)} product info documents")
        
        return {
            **state,
            "product_info_docs": product_info_docs,
            "product_info_context": product_info_context
        }
    
    except Exception as e:
        print(f"⚠️ Could not retrieve product info: {e}")
        return {
            **state,
            "product_info_docs": [],
            "product_info_context": ""
        }


# Node 4: Combine Contexts
def combine_contexts_node(state: RMProposalState) -> RMProposalState:
    """Combine all contexts"""
    print("🔗 [COMBINING] Merging all contexts...")
    
    sections = []
    
    # Web search results
    sections.append("=== WEB SEARCH RESULTS ===\n\n" + state['web_context'])
    
    # Suggested loan products
    if state['suggested_loan_products']:
        products_list = "\n".join([f"- {p}" for p in state['suggested_loan_products']])
        sections.append(f"=== SUGGESTED LOAN PRODUCTS ===\n\n{products_list}")
    
    # Product information sheets
    if state['product_info_context']:
        sections.append("=== LOAN PRODUCT INFORMATION SHEETS ===\n\n" + state['product_info_context'])
    
    combined_context = "\n\n".join(sections)
    
    print("✓ Contexts combined")
    
    return {
        **state,
        "combined_context": combined_context
    }


# Node 5: Generate Analysis with Eligibility Check
def generate_analysis_node(state: RMProposalState) -> RMProposalState:
    """Generate the RM proposal analysis with eligibility assessment"""
    print("🤖 [GENERATING] Creating analysis with eligibility check...\n")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior Relationship Manager (RM) analyst for corporate banking in Malaysia.

        Your task: Based on web search results, recommended loan products, and product information sheets, create a comprehensive RM proposal with ELIGIBILITY ASSESSMENT.

        Structure your analysis as follows:

        ## EXECUTIVE SUMMARY
        Brief overview of the company and key findings

        ## COMPANY ANALYSIS (from web sources)
        - Recent developments and activities
        - Financial health indicators
        - Growth trajectory and expansion plans
        - Specific financing needs identified

        ## RECOMMENDED LOAN PRODUCTS
        For each suggested product, provide:
        - Product name and type
        - Why this product is suitable (based on company needs)
        - Key features relevant to this customer

        ## ELIGIBILITY ASSESSMENT ✓✗
        For EACH recommended loan product, explicitly check against product info sheet criteria:

        ### [Product Name]
        **Eligibility Criteria (from Product Info Sheet):**
        - List each criterion (e.g., minimum revenue, credit score, years in business, industry, collateral requirements, etc.)

        **Assessment Based on Company Profile (from Web Sources):**
        - For EACH criterion, assess whether the company is LIKELY TO MEET ✓ or UNLIKELY TO MEET ✗ based on publicly available information
        - Provide reasoning (e.g., "Company revenue ~RM 850M based on recent reports, exceeds minimum RM 5M requirement ✓")
        - Clearly state "Information not publicly available" for criteria that cannot be assessed from web sources

        **Overall Eligibility:** LIKELY ELIGIBLE / NEEDS VERIFICATION / UNLIKELY TO QUALIFY
        **Information Gaps:** List criteria that require internal verification (e.g., credit score, internal financials)

        ## LOAN OPPORTUNITY ASSESSMENT
        - Estimated loan size and urgency
        - Specific use cases for each product
        - Expected benefits to the customer

        ## RM PROPOSAL STRATEGY
        - Recommended approach for likely eligible products
        - Information to gather for verification (credit score, financials, etc.)
        - Alternative products if primary choices may not be suitable
        - Key decision-makers to target

        ## RISK FACTORS
        - Potential concerns from public information
        - Market or sector risks
        - Eligibility risks that need internal verification

        ## NEXT STEPS
        Concrete actions for the RM team, including:
        - Internal data to verify
        - Documents to request from customer
        - Follow-up actions

        CRITICAL REQUIREMENTS:
        - Use [Web Source N] for web information
        - Use [Product Info N] for product eligibility criteria  
        - Base eligibility assessment on PUBLICLY AVAILABLE information from web sources
        - Be EXPLICIT when information is not available publicly
        - Do not assume internal customer data - only assess what can be determined from web sources
        - Clearly distinguish between confirmed from public sources vs requires verification"""),
                ("user", """Company: {company_name}

        Context:
        {context}

        Please provide a comprehensive RM proposal analysis with detailed eligibility assessment.""")
    ])
    
    chain = prompt | llm_model | StrOutputParser()
    
    try:
        analysis = chain.invoke({
            "company_name": state['company_name'],
            "context": state['combined_context']
        })
        
        print("✓ Analysis with eligibility check generated")
        
        return {
            **state,
            "analysis": analysis
        }
    
    except Exception as e:
        print(f"⚠️ Analysis generation error: {e}")
        return {
            **state,
            "analysis": "",
            "error": f"Analysis generation failed: {str(e)}"
        }


# Node 6: Save Results
def save_results_node(state: RMProposalState) -> RMProposalState:
    """Save the analysis to a file"""
    print("💾 [SAVING] Writing to file...")
    
    company_name = state['company_name']
    analysis = state['analysis']
    web_results = state['web_results']

    output_dir = "C:/Users/noeln/OneDrive/Desktop/Agentic RAG/generate-personalised-rm-proposals/2. output"  # or an absolute path like "C:/Users/Noel/Documents/BankReports"
    os.makedirs(output_dir, exist_ok=True)  # ensures the directory exists
    filename = os.path.join(output_dir, f"{company_name.replace(' ', '_')}_eligibility_analysis.txt")
    # filename = f"{company_name.replace(' ', '_')}_eligibility_analysis.txt"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"RM PROPOSAL WITH ELIGIBILITY ANALYSIS: {company_name}\n")
            f.write("="*80 + "\n")
            f.write("(Web Search + Product Info Sheets)\n")
            f.write("="*80 + "\n\n")
            
            # Suggested products
            if state['suggested_loan_products']:
                f.write("🎯 SUGGESTED LOAN PRODUCTS:\n")
                for i, product in enumerate(state['suggested_loan_products'], 1):
                    f.write(f"   {i}. {product}\n")
                f.write("\n" + "="*80 + "\n\n")
            
            f.write(analysis)
            f.write("\n\n" + "="*80 + "\n")
            
            # Web sources
            f.write("\n📰 WEB SOURCES:\n")
            for i, source in enumerate(web_results, 1):
                f.write(f"[Web Source {i}] {source['title']}\n")
                f.write(f"               {source['url']}\n\n")
            
            # Product info sheets
            if state['product_info_docs']:
                f.write("\n📋 PRODUCT INFORMATION SHEETS:\n")
                for i, doc in enumerate(state['product_info_docs'], 1):
                    source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    f.write(f"[Product Info {i}] {source}\n")
        
        print(f"✅ Eligibility analysis saved to {filename}")
        
        return state
    
    except Exception as e:
        print(f"⚠️ Save error: {e}")
        return {
            **state,
            "error": f"Save failed: {str(e)}"
        }


# Build the graph
def create_rm_proposal_graph():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(RMProposalState)
    
    # Add nodes
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("identify_loan_products", identify_loan_products_node)
    workflow.add_node("retrieve_product_info", retrieve_product_info_node)
    workflow.add_node("combine_contexts", combine_contexts_node)
    workflow.add_node("generate_analysis", generate_analysis_node)
    workflow.add_node("save_results", save_results_node)
    
    # Define the flow
    workflow.set_entry_point("web_search")
    workflow.add_edge("web_search", "identify_loan_products")
    workflow.add_edge("identify_loan_products", "retrieve_product_info")
    workflow.add_edge("retrieve_product_info", "combine_contexts")
    workflow.add_edge("combine_contexts", "generate_analysis")
    workflow.add_edge("generate_analysis", "save_results")
    workflow.add_edge("save_results", END)
    
    return workflow.compile()


# Main function to run the workflow
def create_hybrid_rm_proposal_analysis(
    company_name: str, 
    web_query: str,
    use_vectorstore: bool = True
):
    """
    Generate RM proposal analysis with product eligibility check
    
    Args:
        company_name: Name of the company to analyze
        web_query: Query for web search (Tavily)
        use_vectorstore: Whether to include internal document search
    """
    
    # Create the graph
    app = create_rm_proposal_graph()
    
    # Initial state
    initial_state = {
        "company_name": company_name,
        "web_query": web_query,
        "use_vectorstore": use_vectorstore,
        "web_results": [],
        "web_context": "",
        "suggested_loan_products": [],
        "product_info_docs": [],
        "product_info_context": "",
        "combined_context": "",
        "analysis": "",
        "error": ""
    }
    
    # Run the workflow
    print("="*80)
    print(f"🚀 STARTING RM ELIGIBILITY ANALYSIS: {company_name}")
    print("="*80 + "\n")
    
    final_state = app.invoke(initial_state)
    
    return (
        final_state['analysis'], 
        final_state['web_results'], 
        final_state['suggested_loan_products'],
        final_state['product_info_docs']
    )


# Example Usage
if __name__ == "__main__":
    
    # Option 1: First-time setup - Load documents into vectorstore
    # Make sure your vectorstore includes:
    # - Loan product information sheets (with eligibility criteria)
    """
    print("="*80)
    print("INITIAL SETUP: Loading Internal Documents")
    print("="*80)
    rag_system.create_vectorstore(directory_path="./my_documents")
    print("\n")
    """
    
    # Option 2: Run eligibility analysis with LangGraph (Dynamic Input)
    print("="*80)
    print("RM PROPOSAL ELIGIBILITY ANALYSIS SYSTEM")
    print("="*80)
    print("\nEnter company details for analysis:\n")
    
    # Get dynamic input from user
    company = input("Company Name: ").strip()
    
    if not company:
        print("❌ Error: Company name cannot be empty")
        exit(1)
    
    # Auto-generate web query based on company name
    web_query = f"{company} Malaysia news financial performance expansion plans 2024 2025"
    
    print(f"\n🔍 Web Search Query: {web_query}")
    
    # Optional: Ask if user wants to customize the web query
    customize = input("\nDo you want to customize the web search query? (y/n): ").strip().lower()
    if customize == 'y':
        custom_query = input("Enter custom web query: ").strip()
        if custom_query:
            web_query = custom_query
    
    # Optional: Ask if user wants to use vectorstore
    use_vs = input("\nSearch internal documents (vectorstore)? (y/n, default: y): ").strip().lower()
    use_vectorstore = use_vs != 'n'
    
    print("\n" + "="*80)
    print("Starting analysis...")
    print("="*80 + "\n")
    
    # Run analysis
    analysis, web_sources, loan_products, product_docs = create_hybrid_rm_proposal_analysis(
        company_name=company,
        web_query=web_query,
        use_vectorstore=use_vectorstore
    )
    
    # Display results
    print("\n" + "="*80)
    print(f"RM ELIGIBILITY ANALYSIS: {company}")
    print("="*80)
    print(analysis)
    print("\n" + "="*80)
    
    # Show sources
    print(f"\n📰 Web Sources: {len(web_sources)}")
    print(f"🎯 Suggested Loan Products: {len(loan_products)}")
    for i, product in enumerate(loan_products, 1):
        print(f"   {i}. {product}")
    print(f"📋 Product Info Documents: {len(product_docs)}")
    
    # Ask if user wants to analyze another company
    print("\n" + "="*80)
    another = input("\nAnalyze another company? (y/n): ").strip().lower()
    if another == 'y':
        print("\n")
        # Restart by running the script again
        import sys
        os.execv(sys.executable, ['python'] + sys.argv)
