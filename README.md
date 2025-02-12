### Set up

#### 1. Installation
This code requires Python >= 3.9.


```
pip install -r requirements.txt
```

#### 2. Environment Variables

Create a file named .env and add the following lines, replacing placeholders with your actual values:

```
MONGODB_URI=
EMBEDDING_MODEL=
DB_NAME=
DB_COLLECTION=
GEMINI_KEY=
```

- MONGODB_URI: URI of your MongoDB Atlas instance.
- EMBEDDING_MODEL: Name of the embedding model you're using for text embedding.
- DB_NAME: Name of the database in your MongoDB Atlas.
- DB_COLLECTION: Name of the collection within the database.
- GEMINI_KEY: Your key to access the Gemini API.



#### 4. Edit your Prompt in serve.py

In the serve.py file, you can see that we used the prompt like this. This prompt was enhanced by adding information about your products to it.

```
f"Hãy trở thành chuyên gia tư vấn dịch vụ của ngân hàng. Câu hỏi của khách hàng: {query}\nTrả lời câu hỏi dựa vào các thông tin dưới đây: {source_information}."
```

- query: Query from the user.
- source_information: Information we get from our database.



#### 5. Run server

```
python serve.py
```
