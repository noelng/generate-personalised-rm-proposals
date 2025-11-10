# Summary
1. generate-personalised-rm-proposals.py
   - The first version (websearch -> retrieve data from vectorstore -> combine context -> generate analysis)
2. langgraph_rm_proposal_v1.py
   - Same content as the above but added langgraph to orchestrate the workflow 
3. langgraph_rm_proposal_v2.py
   - Improve internal query for vectorstore to be dynamically generated based on the web search results, specifically focusing on loan opportunity assessment 
      -- Analyze web results to identify which loan products might be suitable
      -- Search vectorstore for product info sheets and check customer eligibility criteria
   - Updated the script to accept dynamic input
   - Dynamic Company Input: Prompts user to enter company name
   - Auto-generated Web Query: Creates a sensible default query
   - Optional Query Customization: User can modify the web search query
   - Optional Vectorstore Toggle: User can choose to skip internal docs
   - Loop for Multiple Companies: Can analyze another company without restarting
   - Input Validation: Checks for empty company name



# Workflow: Eligibility-Focused Process
## Flow Diagram

1. Web Search (company info)
   ↓
2. Identify Loan Products (AI analyzes web results → suggests products)
   ↓
3. Retrieve Product Info Sheets (search vectorstore for eligibility criteria)
   ↓
4. Combine Contexts
   ↓
5. Generate Analysis (with eligibility assessment based on public data)
   ↓
6. Save Results