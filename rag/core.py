# import pymongo
# import google.generativeai as genai
# from IPython.display import Markdown
# import textwrap
# from embeddings import SentenceTransformerEmbedding, EmbeddingConfig

# class RAG():
#     def __init__(self, 
#             mongodbUri: str,
#             dbName: str,
#             dbCollection: str,
#             llm,
#             embeddingName: str ='keepitreal/vietnamese-sbert',
#         ):
#         self.client = pymongo.MongoClient(mongodbUri)
#         self.db = self.client[dbName] 
#         self.collection = self.db[dbCollection]
#         self.embedding_model = SentenceTransformerEmbedding(
#             EmbeddingConfig(name=embeddingName)
#         )
#         self.llm = llm

#     def get_embedding(self, text):
#         if not text.strip():
#             return []

#         embedding = self.embedding_model.encode(text)
#         return embedding.tolist()

#     def vector_search(
#             self, 
#             user_query: str, 
#             limit=4):
#         """
#         Perform a vector search in the MongoDB collection based on the user query.

#         Args:
#         user_query (str): The user's query string.

#         Returns:
#         list: A list of matching documents.
#         """

#         # Generate embedding for the user query
#         query_embedding = self.get_embedding(user_query)

#         if query_embedding is None:
#             return "Invalid query or embedding generation failed."

#         # Define the vector search pipeline
#         vector_search_stage = {
#             "$vectorSearch": {
#                 "index": "vector_index",
#                 "queryVector": query_embedding,
#                 "path": "embedding",
#                 "numCandidates": 400,
#                 "limit": limit,
#             }
#         }

#         unset_stage = {
#             "$unset": "embedding" 
#         }

#         project_stage = {
#             "$project": {
#                 "_id": 0,  
#                 "title": 1, 
#                 # "product_specs": 1,
#                 "color_options": 1,
#                 "current_price": 1,
#                 "product_promotion": 1,
#                 "score": {
#                     "$meta": "vectorSearchScore"
#                 }
#             }
#         }

#         pipeline = [vector_search_stage, unset_stage, project_stage]

#         # Execute the search
#         results = self.collection.aggregate(pipeline)

#         return list(results)

#     def enhance_prompt(self, query):
#         get_knowledge = self.vector_search(query, 10)
#         enhanced_prompt = ""
#         i = 0
#         for result in get_knowledge:
#             if result.get('current_price'):
#                 i += 1
#                 enhanced_prompt += f"\n {i}) Tên: {result.get('title')}"
                
#                 if result.get('current_price'):
#                     enhanced_prompt += f", Giá: {result.get('current_price')}"
#                 else:
#                     # Mock up data
#                     # Retrieval model pricing from the internet.
#                     enhanced_prompt += f", Giá: Liên hệ để trao đổi thêm!"
                
#                 if result.get('product_promotion'):
#                     enhanced_prompt += f", Ưu đãi: {result.get('product_promotion')}"
#         return enhanced_prompt

#     def generate_content(self, prompt):
#         return self.llm.generate_content(prompt)

#     def _to_markdown(text):
#         text = text.replace('•', '  *')
#         return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


import pymongo
import google.generativeai as genai
from IPython.display import Markdown
import textwrap
from embeddings import SentenceTransformerEmbedding, EmbeddingConfig
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# Tải các dữ liệu cần thiết cho nltk (nếu chưa có)
nltk.download('punkt')

class RAG():
    def __init__(self, 
            mongodbUri: str,
            dbName: str,
            dbCollection: str,
            llm,
            embeddingName: str ='dangvantuan/vietnamese-embedding',
        ):
        self.client = pymongo.MongoClient(mongodbUri)
        self.db = self.client[dbName] 
        self.collection = self.db[dbCollection]
        self.embedding_model = SentenceTransformerEmbedding(
            EmbeddingConfig(name=embeddingName)
        )
        self.llm = llm
        self.bm25_model = self._prepare_bm25_model()

    def _prepare_bm25_model(self):
        # Lấy toàn bộ dữ liệu từ MongoDB
        documents = self.collection.find({}, {"_id": 0, "text": 1})
        corpus = [doc['text'] for doc in documents if 'text' in doc]

        # Tokenize dữ liệu
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]

        # Tạo mô hình BM25
        bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = corpus  # Lưu để truy xuất văn bản sau
        return bm25

    def bm25_search(self, query, top_k=3):
        # Tokenize truy vấn
        tokenized_query = word_tokenize(query.lower())

        # Tính điểm BM25
        scores = self.bm25_model.get_scores(tokenized_query)

        # Lấy top_k kết quả
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Trả về văn bản và điểm
        results = [{"text": self.bm25_corpus[i], "score": scores[i]} for i in top_indices]
        return results

    def get_embedding(self, text):
        if not text.strip():
            return []

        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def vector_search(self, user_query: str, limit=4):
        """
        Perform a vector search in the MongoDB collection based on the user query.
        """
        query_embedding = self.get_embedding(user_query)
        if query_embedding is None:
            return "Invalid query or embedding generation failed."

        # Define the vector search pipeline
        vector_search_stage = {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 399,
                "limit": limit,
            }
        }

        unset_stage = {
            "$unset": "embedding" 
        }

        project_stage = {
            "$project": {
                "_id": 0,  
               "text": 1,
                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }

        pipeline = [vector_search_stage, unset_stage, project_stage]
        results = self.collection.aggregate(pipeline)

        return list(results)

    def hybrid_search(self, query, top_k=3):
        # Thực hiện cả BM25 và vector search
        bm25_results = self.bm25_search(query, top_k)
        vector_results = self.vector_search(query, top_k)

        # Kết hợp kết quả từ BM25 và vector search
        combined_results = bm25_results + vector_results

        # Loại bỏ trùng lặp theo văn bản
        seen_texts = set()
        unique_results = []
        for result in combined_results:
            if result['text'] not in seen_texts:
                unique_results.append(result)
                seen_texts.add(result['text'])

        # Sắp xếp lại theo điểm số (nếu cần)
        unique_results = sorted(unique_results, key=lambda x: x.get('score', 0), reverse=True)
        return unique_results[:top_k]

    def enhance_prompt(self, query):
        # Lấy dữ liệu từ hybrid_search
        get_knowledge = self.hybrid_search(query, 3)
        
        # Tạo prompt nâng cao
        enhanced_prompt = ""
        i = 0
        
        for result in get_knowledge:
            i += 1
            enhanced_prompt += f"\n {i}) Nội dung: {result.get('text')}"
        
        return enhanced_prompt

    def generate_content(self, prompt):
        return self.llm.generate_content(prompt)

    @staticmethod
    def _to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))