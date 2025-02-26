# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.document_loaders.image import UnstructuredImageLoader

# from dotenv import load_dotenv

# load_dotenv()
# import os
# google_api = os.getenv("google_api")

# def setup_coordinator(query):
#     """Setup the main agent for answering questions about Muhammad Bahjat."""
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash-exp",
#         google_api_key=google_api,
#         temperature=0.5
#     )

#     # Define the prompt template with system message
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """Answer the question """),
#         MessagesPlaceholder(variable_name="history"),
#         ("human", "{input}")
#     ])

#     # Create the chain
#     chain = prompt | llm

#     # Add conversation history
#     chain_with_history = RunnableWithMessageHistory(
#         chain,
#         lambda session_id: ChatMessageHistory(),
#         input_messages_key="input",
#         history_messages_key="history"
#     )
    
#     try:
#         response = chain_with_history.invoke(
#             {"input": query},
#             config={
#                 "configurable": {
#                     "session_id": "bahjat_assistant"
#                 }
#             }
#         )
#         return response.content
#     except Exception as e:
#         return f"An error occurred: {str(e)}"

# # Example usage
# if __name__ == "__main__":
#     image_path = "4.jpg"
#     loader = UnstructuredImageLoader(image_path)
#     data = loader.load()
#     data[0]
#     print(data[0])
#     response = setup_coordinator(f"whats in this image? {image_path}")
#     print(f"Final response: {response}")




