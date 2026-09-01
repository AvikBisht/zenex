import os
import json
import asyncio
from typing import TypedDict, List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize FastAPI app for the backend
app = FastAPI(title="Oracle Desk - Multi-Agent AI Backend")

# Initialize LLM (Use GPT-4o-mini or a fast open-source model via Groq for low latency)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# --- 1. DEFINE THE GRAPH STATE ---
# This dictionary tracks the data as it moves through the agents
class AgentState(TypedDict):
    ticker: str
    user_profile: str  # 'conservative', 'moderate', 'aggressive'
    market_data: Dict[str, Any]
    technical_signal: Optional[Dict[str, Any]]
    fundamental_signal: Optional[Dict[str, Any]]
    sentiment_signal: Optional[Dict[str, Any]]
    final_recommendation: Optional[Dict[str, Any]]
    degraded_mode: bool
    errors: List[str]

# --- 2. AGENT 1: TECHNICAL & PRICE DYNAMICS ---
def technical_agent(state: AgentState):
    print("Executing Technical Agent...")
    try:
        data = state["market_data"]
        # In a real app, you would fetch real-time NSE data here.
        # We pass the numeric data to the LLM to generate a reasoned analysis.
        prompt = f"""
        Analyze this market data for {state['ticker']}:
        5-Day Momentum: {data.get('momentum5d')}%
        Volume Ratio: {data.get('volumeRatio')}x average
        
        Provide a JSON response with:
        "signal": "BUY", "SELL", or "HOLD"
        "confidence": 0-100
        "reasoning": "1 sentence explanation"
        """
        response = llm.invoke(prompt)
        
        # Parse LLM output (assuming JSON output configured)
        parsed = json.loads(response.content)
        return {"technical_signal": parsed}
    except Exception as e:
        # Graceful Degradation: If feed fails, flag it but don't crash
        return {"technical_signal": {"signal": "UNAVAILABLE", "confidence": 0, "reasoning": "Technical feed timeout."}, "degraded_mode": True, "errors": state.get("errors", []) + [str(e)]}

# --- 3. AGENT 2: FUNDAMENTAL & RAG (VECTOR DB) ---
def fundamental_rag_agent(state: AgentState):
    print("Executing Fundamental RAG Agent...")
    try:
        # Connect to Vector DB (Chroma) containing SEBI filings & Earnings Call Transcripts
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
        retriever = vector_db.as_retriever(search_kwargs={"k": 2})
        
        # Retrieve actual documents based on ticker
        docs = retriever.invoke(f"Recent financial guidance and risk factors for {state['ticker']}")
        context = "\n".join([f"Source: {d.metadata['title']} | Content: {d.page_content}" for d in docs])
        
        prompt = f"""
        Based on the following regulatory filings for {state['ticker']}:
        {context}
        
        Evaluate the fundamental health. Provide a JSON response with:
        "signal": "CONSTRUCTIVE", "CAUTIONARY", or "MIXED"
        "confidence": 0-100
        "reasoning": "1 sentence citing the specific source document."
        """
        response = llm.invoke(prompt)
        parsed = json.loads(response.content)
        return {"fundamental_signal": parsed}
    except Exception as e:
        return {"fundamental_signal": {"signal": "UNAVAILABLE", "confidence": 0, "reasoning": "RAG database unreachable."}, "degraded_mode": True}

# --- 4. AGENT 3: MACRO & SENTIMENT ---
def sentiment_agent(state: AgentState):
    print("Executing Sentiment Agent...")
    # Simulating fetching news/Twitter sentiment
    return {
        "sentiment_signal": {
            "signal": "NEUTRAL",
            "confidence": 80,
            "reasoning": "Retail sentiment is flat, no major FII/DII institutional block deals detected today."
        }
    }

# --- 5. SYNTHESIS ORCHESTRATOR & RISK PROFILER ---
def synthesis_agent(state: AgentState):
    print("Executing Synthesis & Risk Profiling...")
    profile = state["user_profile"]
    
    # Pass all agent outputs and the user's exact risk profile to the Orchestrator LLM
    prompt = f"""
    You are the Master Synthesis Agent. Combine these 3 agent inputs for {state['ticker']}:
    1. Technical: {state['technical_signal']}
    2. Fundamental: {state['fundamental_signal']}
    3. Sentiment: {state['sentiment_signal']}
    
    The investor's risk profile is: {profile.upper()}.
    Conservative: Weight fundamentals heavily. Avoid risk.
    Aggressive: Weight momentum heavily. Tolerate risk.
    
    Provide a final JSON decision:
    "recommendation": "STRONG BUY", "BUY", "HOLD", "SELL", or "STRONG SELL"
    "composite_confidence": 0-100
    "reasoning": "2 sentences explaining the decision tailored to the user profile, explicitly resolving any conflicts between the agents."
    """
    response = llm.invoke(prompt)
    parsed = json.loads(response.content)
    
    return {"final_recommendation": parsed}

# --- 6. BUILD THE LANGGRAPH WORKFLOW ---
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("tech", technical_agent)
workflow.add_node("fund", fundamental_rag_agent)
workflow.add_node("sent", sentiment_agent)
workflow.add_node("synth", synthesis_agent)

# Define parallel execution (Tech, Fund, and Sent run at the same time)
workflow.set_entry_point("tech") # In LangGraph, we can fan-out.
# For simplicity in script, standard fan-out:
workflow.add_edge("tech", "synth")
workflow.add_edge("fund", "synth")
workflow.add_edge("sent", "synth")
workflow.add_edge("synth", END)

# Compile graph
app_graph = workflow.compile()

# --- 7. FASTAPI ENDPOINT (For your HTML UI) ---
class AnalysisRequest(BaseModel):
    ticker: str
    user_profile: str
    market_data: Dict[str, Any]

@app.post("/api/analyze")
async def run_analysis(req: AnalysisRequest):
    initial_state = {
        "ticker": req.ticker,
        "user_profile": req.user_profile,
        "market_data": req.market_data,
        "degraded_mode": False,
        "errors": []
    }
    
    # Run the graph
    result = app_graph.invoke(initial_state)
    
    # Return to HTML frontend
    return {
        "agents": {
            "technical": result["technical_signal"],
            "fundamental": result["fundamental_signal"],
            "sentiment": result["sentiment_signal"]
        },
        "synthesis": result["final_recommendation"],
        "degraded_mode": result["degraded_mode"]
    }
