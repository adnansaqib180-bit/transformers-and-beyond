from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text_generation'
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('how many goals cristiano ronaldo has scored')
print(result.content)

# ====== for embedding models ========

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpointEmbeddings 
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
)


result = llm.embed_query('how many goals cristiano ronaldo has scored')
print(len(result))

# embedding multiple sentences 

document = ['sentence1','sentence2','sentence3']
result = llm.embed_documents(document)

