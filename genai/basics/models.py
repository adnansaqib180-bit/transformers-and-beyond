from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')
result = model.invoke('what is the capital of pakistan ?')

print(result.content)

#====================================
# open ai 

from langchain_openai import OpenAI

model = OpenAI(model='gpt-4')
model.invoke(input='what is ...')

#  ++====================
# gmeinai / google 

from langchain_google_genai import ChatGoogleGenerativeAI

model =  ChatGoogleGenerativeAI(model='geminie3.12')
model.invoke(input='what ...')

# same with all other models and companies jus diffrent pacj=kage name althogh the sytax is same 

