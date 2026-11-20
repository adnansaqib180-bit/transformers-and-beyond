# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
# from dotenv import load_dotenv
# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
#     task='text_generation'
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke('how many goals cristiano ronaldo has scored')
# print(result.content)

# # ====== for embedding models ========

# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpointEmbeddings 
# from dotenv import load_dotenv
# load_dotenv()

# llm = HuggingFaceEndpointEmbeddings(
#     repo_id="sentence-transformers/all-MiniLM-L6-v2",
# )


# result = llm.embed_query('how many goals cristiano ronaldo has scored')
# print(len(result))

# embedding multiple sentences 

# document = ['sentence1','sentence2','sentence3']
# result = llm.embed_documents(document)

# # saaving models on to local machine 


from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation',
    pipeline_kwargs={
        'tempreture':0.3,
        'max_new_tokens':100,
        'return_full-text' : False,
        'do_sample':True

    }
    ) 

model = ChatHuggingFace(llm=llm)
result = model.invoke('how many goals cristiano has scored?')

print(result.content)