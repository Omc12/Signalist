"""
RAG Chat Route
Allows users to chat with the AI about specific stocks using RAG context.
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

# Data models
class ChatRequest(BaseModel):
    ticker: str
    message: str
    history: Optional[List[Dict[str, str]]] = []

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, str]]] = []

@router.post("/chat", response_model=ChatResponse)
async def chat_with_stock(request: ChatRequest):
    """
    Chat with AI about a specific stock using real-time news context (RAG).
    """
    try:
        # Check API Keys
        newsdata_key = os.getenv("NEWSDATA_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        
        if not gemini_key or gemini_key == "your_gemini_api_key_here":
            return ChatResponse(
                response="I'm sorry, but I can't analyze this stock right now because the AI service is not configured. Please add a valid GEMINI_API_KEY to the backend .env file.",
                sources=[]
            )

        # 1. Fetch News Context if available
        context_text = ""
        sources = []
        
        if newsdata_key and newsdata_key != "your_newsdata_api_key_here":
            try:
                # Map ticker to company name
                ticker_map = {
                    "RELIANCE": "Reliance Industries",
                    "TCS": "Tata Consultancy Services",
                    "INFY": "Infosys",
                    "HDFC": "HDFC Bank",
                    "ICICIBANK": "ICICI Bank",
                    "SBIN": "State Bank of India",
                    "WIPRO": "Wipro"
                }
                company = ticker_map.get(request.ticker.split('.')[0], request.ticker)
                
                # Fetch news
                response = requests.get(
                    "https://newsdata.io/api/1/news",
                    params={"apikey": newsdata_key, "q": company, "language": "en"},
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    articles = data.get("results", [])[:5]
                    
                    if articles:
                        context_pieces = []
                        for article in articles:
                            title = article.get('title', 'No Title')
                            desc = article.get('description', '') or ''
                            source = article.get('source_id', 'News')
                            link = article.get('link', '#')
                            
                            context_pieces.append(f"Title: {title}\nSummary: {desc[:200]}")
                            sources.append({"title": title, "link": link, "source": source})
                        
                        context_text = "Here is the recent news context:\n" + "\n---\n".join(context_pieces)
            except Exception as e:
                print(f"News fetch failed: {e}")
        
        # 2. Build Prompt
        system_prompt = f"""You are Signalist AI, a financial assistant. 
User is asking about {request.ticker}.
Answer their question based on the news context provided below. 
If the news doesn't contain the answer, use your general knowledge but mention that it might not be up-to-the-minute.
Be concise, professional, and data-driven.
Avoid financial advice disclaimers in every sentence, but maintain a neutral tone.

{context_text}
"""
        
        # 3. Call Gemini
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{system_prompt}\n\nUser Question: {request.message}")
        
        return ChatResponse(
            response=response.text,
            sources=sources
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
