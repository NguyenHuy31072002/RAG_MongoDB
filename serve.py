# Sử dụng FastAPI  
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import google.generativeai as genai
from rag.core import RAG
from embeddings import OpenAIEmbedding
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
import openai
from reflection import Reflection
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Load environment variables
load_dotenv()
MONGODB_URI = os.getenv('MONGODB_URI')
DB_NAME = os.getenv('DB_NAME')
DB_COLLECTION = os.getenv('DB_COLLECTION')
LLM_KEY = os.getenv('GEMINI_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'

class EmbeddingsWrapper:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def encode(self, texts):
        return self.embeddings.embed_documents(texts)

# Initialize embeddings
base_embeddings = SentenceTransformerEmbeddings(model_name='dangvantuan/vietnamese-embedding')
embeddings = EmbeddingsWrapper(base_embeddings)

# Semantic Router Setup
PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'

productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(embeddings, routes=[productRoute, chitchatRoute])

# Set up LLMs
genai.configure(api_key=LLM_KEY)
llm = genai.GenerativeModel('gemini-1.5-pro')
reflection = Reflection(llm=llm)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể thay đổi thành domain cụ thể cho production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG
rag = RAG(
    mongodbUri=MONGODB_URI,
    dbName=DB_NAME,
    dbCollection=DB_COLLECTION,
    embeddingName='keepitreal/vietnamese-sbert',
    llm=llm,
)

def process_query(query: str) -> str:
    return query.lower()

# Define request model
class MessagePart(BaseModel):
    text: str

class Message(BaseModel):
    role: str = None
    parts: List[MessagePart]

class QueryRequest(BaseModel):
    messages: List[Dict[str, Any]]

@app.post("/api/search")
async def handle_query(request: List[Dict[str, Any]]):
    try:
        data = list(request)
        query = data[-1]["parts"][0]["text"]
        query = process_query(query)

        if not query:
            raise HTTPException(status_code=400, detail="No query provided")
        
        guidedRoute = semanticRouter.guide(query)[1]

        if guidedRoute == PRODUCT_ROUTE_NAME:
            print("Guide to RAGs")
            reflected_query = reflection(data)
            query = reflected_query
            source_information = rag.enhance_prompt(query).replace('<br>', '\n')
            combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
            
            data.append({
                "role": "user",
                "parts": [
                    {
                        "text": combined_information,
                    }
                ]
            })
            response = rag.generate_content(data)
        else:
            print("Guide to LLMs")
            response = llm.generate_content(data)

        return {
            'parts': [
                {
                    'text': response.text,
                }
            ],
            'role': 'model'
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)






# from flask import Flask, request, jsonify
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# from flask_cors import CORS
# from rag.core import RAG
# from embeddings import OpenAIEmbedding
# from semantic_router import SemanticRouter, Route
# from semantic_router.samples import productsSample, chitchatSample
# # from google import genai 
# import openai
# from reflection import Reflection
# from langchain_community.embeddings import SentenceTransformerEmbeddings

# # Load environment variables from .env file
# load_dotenv()
# # Access the key
# MONGODB_URI = os.getenv('MONGODB_URI')
# DB_NAME = os.getenv('DB_NAME')
# DB_COLLECTION = os.getenv('DB_COLLECTION')
# LLM_KEY = os.getenv('GEMINI_KEY')
# EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'


# class EmbeddingsWrapper:
#     def __init__(self, embeddings):
#         self.embeddings = embeddings
    
#     def encode(self, texts):
#         # SentenceTransformerEmbeddings sử dụng embed_documents thay vì encode
#         return self.embeddings.embed_documents(texts)



# # Thay đổi cách khởi tạo embeddings:
# base_embeddings = SentenceTransformerEmbeddings(model_name='dangvantuan/vietnamese-embedding')
# embeddings = EmbeddingsWrapper(base_embeddings) 


# #embeddings = SentenceTransformerEmbeddings(model_name='dangvantuan/vietnamese-embedding')



# # --- Semantic Router Setup --- #
# PRODUCT_ROUTE_NAME = 'products'
# CHITCHAT_ROUTE_NAME = 'chitchat'


# productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
# chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
# semanticRouter = SemanticRouter(embeddings, routes=[productRoute, chitchatRoute])

# # --- End Semantic Router Setup --- #


# # --- Set up LLMs --- #

# genai.configure(api_key=LLM_KEY)
# llm = genai.GenerativeModel('gemini-1.5-pro')


# reflection = Reflection(llm=llm)

# # --- End Reflection Setup --- #

# app = Flask(__name__)
# CORS(app)

# # Initialize RAG
# rag = RAG(
#     mongodbUri=MONGODB_URI,
#     dbName=DB_NAME,
#     dbCollection=DB_COLLECTION,
#     embeddingName='keepitreal/vietnamese-sbert',
#     llm=llm,
# )

# def process_query(query):
#     return query.lower()

# @app.route('/api/search', methods=['POST'])
# def handle_query():
#     data = list(request.get_json())

#     query = data[-1]["parts"][0]["text"]

#     query = process_query(query)

#     if not query:
#         return jsonify({'error': 'No query provided'}), 400
    
#     # get last message
    
#     guidedRoute = semanticRouter.guide(query)[1]

#     if guidedRoute == PRODUCT_ROUTE_NAME:
#         # Decide to get new info or use previous info
#         # Guide to RAG system
#         print("Guide to RAGs")

#         reflected_query = reflection(data)

#         # print('====query', query)
#         # print('reflected_query', reflected_query)

#         query = reflected_query
#         source_information = rag.enhance_prompt(query).replace('<br>', '\n')
#         combined_information = f"Hãy trở thành chuyên gia tư vấn bán hàng cho một cửa hàng điện thoại. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin sản phẩm dưới đây: {source_information}."
#         data.append({
#             "role": "user",
#             "parts": [
#                 {
#                     "text": combined_information,
#                 }
#             ]
#         })
#         response = rag.generate_content(data)
#     else:
#         # Guide to LLMs
#         print("Guide to LLMs")
#         response = llm.generate_content(data)

#     # print('====data', data)
    
#     return jsonify({
#         'parts': [
#             {
#             'text': response.text,
#             }
#         ],
#         'role': 'model'
#         })

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5002, debug=True)











