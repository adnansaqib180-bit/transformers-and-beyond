from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model1 = ChatGroq(model='openai/gpt-oss-120b')
model2 = ChatGroq(model='openai/gpt-oss-safeguard-20b')
parser = StrOutputParser()

promt1 = PromptTemplate.from_template('make notes of the following text \n {text}')
promt2 = PromptTemplate.from_template('make 3 question and answer of the following text \n {text}')
promt3 = PromptTemplate.from_template('merge the following notes and quiz \n notes -> \n {notes} \n quiz -> {quiz}')

parallel_chain = RunnableParallel({
    'notes' : promt1 | model1 | parser,
    'quiz' : promt2 | model2 |parser
})
chain = parallel_chain |  promt3 |model1|parser 
text = input('enter your text here : \n')
results = chain.invoke({'text' : text})
print(results)
chain.get_graph().print_ascii()