# Sử dụng Gemini
class Reflection:
    def __init__(self, llm):
        self.llm = llm

    def _concat_and_format_texts(self, data):
        # Kết hợp và định dạng các đoạn văn bản từ lịch sử chat
        concatenatedTexts = []
        for entry in data:
            role = entry.get('role', '')
            all_texts = ' '.join(part['text'] for part in entry['parts'])
            concatenatedTexts.append(f"{role}: {all_texts} \n")
        return ''.join(concatenatedTexts)

    
    def __call__(self, chatHistory, lastItemsConsidereds=100):
        # Giới hạn số lượng lịch sử chat để xử lý
        if len(chatHistory) >= lastItemsConsidereds:
            chatHistory = chatHistory[len(chatHistory) - lastItemsConsidereds:]

        # Kết hợp và định dạng lịch sử chat
        historyString = self._concat_and_format_texts(chatHistory)

        # Tạo prompt cho mô hình Gemini
        higherLevelSummariesPrompt = f"""
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question in Vietnamese which can be understood without the chat history. 
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is. 
        Chat history: {historyString}
        """

        print(higherLevelSummariesPrompt)

        # Gọi mô hình Gemini để xử lý prompt
        response = self.llm.generate_content(higherLevelSummariesPrompt)

        # Trả về kết quả từ mô hình
        return response.text


# class Reflection():
#     def __init__(self, llm):
#         self.llm = llm

#     def _concat_and_format_texts(self, data):
#         concatenatedTexts = []
#         for entry in data:
#             role = entry.get('role', '')
#             all_texts = ' '.join(part['text'] for part in entry['parts'])
#             concatenatedTexts.append(f"{role}: {all_texts} \n")
#         return ''.join(concatenatedTexts)


#     def __call__(self, chatHistory, lastItemsConsidereds=100):
        
#         if len(chatHistory) >= lastItemsConsidereds:
#             chatHistory = chatHistory[len(chatHistory) - lastItemsConsidereds:]

#         historyString = self._concat_and_format_texts(chatHistory)

#         higherLevelSummariesPrompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question in Vietnamese which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is. {historyString}
#         """.format(historyString=historyString)

#         print(higherLevelSummariesPrompt)

#         completion = self.llm.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": higherLevelSummariesPrompt
#                 }
#             ]
#         )
    
#         return completion.choices[0].message.content

