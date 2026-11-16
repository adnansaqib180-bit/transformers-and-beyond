from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

client = ChatGroq(model= 'openai/gpt-oss-120b')

promt= ChatPromptTemplate.from_messages([
    ('system' ,'you are a customer suppport agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])
chat_history = []
with open('transformers-and-beyond/genai/basics/history.txt','r') as file:
    chat_history.append(file.readline())


promt = promt.invoke({'chat_history':chat_history,'query': 'when will i get my order 1234?'})

print((client.invoke(promt)).content)