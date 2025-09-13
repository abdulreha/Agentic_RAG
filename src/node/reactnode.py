"""Smart Fallback RAG Node - Documents First, Wikipedia as Fallback"""

from typing import List, Optional
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
import re

class RAGNodes:
    """Contains node functions for RAG workflow with smart fallback"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, lang="en")
        self.wiki_tool = WikipediaQueryRun(api_wrapper=self.wiki_wrapper)

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve documents from uploaded content"""
        docs = self.retriever.invoke(state.question)
        print(f"DEBUG: Retrieved {len(docs)} documents from uploaded content")
        
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def is_incomplete_answer(self, answer: str) -> bool:
        """
        Determine if the answer is incomplete or indicates missing information
        """
        incomplete_indicators = [
            "no information",
            "not found",
            "cannot find",
            "not mentioned",
            "not provided",
            "not available",
            "insufficient information",
            "don't have enough",
            "context doesn't contain",
            "not in the documents",
            "need more context",
            "please provide",
            "i need more",
            "cannot answer",
            "unable to answer"
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in incomplete_indicators)

    def answer_with_documents(self, state: RAGState) -> str:
        """
        Generate answer using only uploaded documents
        """
        if not state.retrieved_docs:
            return "No relevant documents found in uploaded content."
        
        # Combine retrieved documents into context
        context_parts = []
        for i, doc in enumerate(state.retrieved_docs, 1):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            source = meta.get("source", meta.get("title", f"Document {i}"))
            context_parts.append(f"[Document {i} - {source}]\n{doc.page_content}")
        
        full_context = "\n\n".join(context_parts)
        
        # Create prompt for document-only answer
        prompt = f"""Answer the user's question based on the provided context from uploaded documents.

CONTEXT FROM UPLOADED DOCUMENTS:
{full_context}

USER QUESTION: {state.question}

INSTRUCTIONS:
- Answer based on the provided context above
- If the context contains the answer, provide a comprehensive response
- If the context doesn't contain enough information, clearly state "The uploaded documents don't contain sufficient information about [topic]."
- Be specific about what information is missing

ANSWER:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error processing documents: {str(e)}"

    def answer_with_wikipedia(self, state: RAGState, doc_answer: str) -> str:
        """
        Use Wikipedia to supplement the document answer
        """
        print("DEBUG: Using Wikipedia fallback for additional information")
        
        # Create a more specific Wikipedia query based on the question
        wiki_query = self.extract_key_terms(state.question)
        
        try:
            # Get Wikipedia content
            wiki_content = self.wiki_tool.run(wiki_query)
            
            # Combine document answer with Wikipedia info
            combined_prompt = f"""You are answering a question where the user's uploaded documents had limited information.

ORIGINAL QUESTION: {state.question}

ANSWER FROM UPLOADED DOCUMENTS:
{doc_answer}

ADDITIONAL WIKIPEDIA INFORMATION:
{wiki_content}

INSTRUCTIONS:
- First acknowledge what was found in the uploaded documents
- Then supplement with relevant Wikipedia information
- Clearly distinguish between information from documents vs. Wikipedia
- Provide a comprehensive answer combining both sources
- Format as: "Based on your uploaded documents: [doc info]. Additionally, from general knowledge: [wiki info]."

FINAL ANSWER:"""

            response = self.llm.invoke(combined_prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"{doc_answer}\n\nNote: Could not retrieve additional information from Wikipedia due to error: {str(e)}"

    def extract_key_terms(self, question: str) -> str:
        """
        Extract key terms from the question for better Wikipedia search
        """
        # Remove common question words and extract main terms
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 'is', 'are', 'do', 'does', 'can', 'will', 'would', 'should'}
        
        # Clean and split the question
        words = re.findall(r'\b\w+\b', question.lower())
        key_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Return the original question if we can't extract good terms
        return ' '.join(key_words[:5]) if key_words else question

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Smart fallback approach: Try documents first, then Wikipedia if needed
        """
        print(f"DEBUG: Starting smart fallback approach with {len(state.retrieved_docs)} retrieved docs")
        
        # Step 1: Try to answer with uploaded documents
        doc_answer = self.answer_with_documents(state)
        print(f"DEBUG: Document answer: {doc_answer[:100]}...")
        
        # Step 2: Check if the answer is incomplete
        if self.is_incomplete_answer(doc_answer):
            print("DEBUG: Document answer incomplete, trying Wikipedia fallback")
            final_answer = self.answer_with_wikipedia(state, doc_answer)
        else:
            print("DEBUG: Document answer sufficient, using document-only response")
            final_answer = doc_answer
        
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=final_answer
        )

    # Alternative: Full ReAct Agent approach (if you want to switch back)
    def generate_answer_with_react_agent(self, state: RAGState) -> RAGState:
        """
        Alternative: Full ReAct agent with smart tool selection
        """
        def document_tool(query: str) -> str:
            """Primary tool: Use uploaded documents"""
            if not state.retrieved_docs:
                return "No documents available from uploads."
            
            context_parts = []
            for i, doc in enumerate(state.retrieved_docs, 1):
                meta = doc.metadata if hasattr(doc, "metadata") else {}
                source = meta.get("source", f"Document {i}")
                context_parts.append(f"[{source}]\n{doc.page_content[:800]}")
            
            return "UPLOADED DOCUMENTS:\n\n" + "\n\n".join(context_parts)

        def wikipedia_tool(query: str) -> str:
            """Secondary tool: Wikipedia search"""
            try:
                return self.wiki_tool.run(query)
            except Exception as e:
                return f"Wikipedia search failed: {str(e)}"

        # Build tools
        tools = [
            Tool(
                name="uploaded_documents",
                description="Search and retrieve information from the user's uploaded documents. ALWAYS try this FIRST.",
                func=document_tool,
            ),
            Tool(
                name="wikipedia_search",
                description="Search Wikipedia for general knowledge. Use this ONLY if uploaded documents don't have sufficient information.",
                func=wikipedia_tool,
            )
        ]

        # System prompt for smart tool usage
        system_prompt = """You are a smart RAG assistant. Follow this strategy:

1. ALWAYS start by using 'uploaded_documents' tool to check user's documents
2. If documents have sufficient info, answer based on them
3. ONLY if documents lack information, use 'wikipedia_search' for additional context
4. Always cite your sources clearly (documents vs Wikipedia)
5. Prioritize user's documents over general knowledge

Be intelligent about tool selection - don't use Wikipedia unless necessary."""

        # Create and run agent
        agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)
        
        result = agent.invoke({
            "messages": [HumanMessage(content=state.question)]
        })

        # Extract answer
        messages = result.get("messages", [])
        answer = "Could not generate answer."
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", "Could not generate answer.")

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer
        )