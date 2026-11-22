print('\n the program is running ..... \n loading import ......')
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')

parser = StrOutputParser()

class FeedBack(BaseModel):
    sentiment : Literal['positive','negative'] = Field('give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

promt1 = PromptTemplate(
    template='give the sentiment of the feedback either positive or negative {feedback} \n {parser}',
    input_variables=['feedback'],
    partial_variables={'parser':parser2.get_format_instructions()}
)
classifier_chain = promt1 | model | parser2

promt2 = PromptTemplate.from_template('genrate a short suitable message for this positive feed back {feedback}')
promt3 = PromptTemplate.from_template('genrate a short suitable message for this negative feed back {feedback}')

branch_chain= RunnableBranch(
    (lambda x : x.sentiment == 'positive' , promt2 | model | parser ),
    (lambda x : x.sentiment == 'negative' , promt3 | model | parser ),
    RunnableLambda(lambda x : 'could not find the sentiment ')
)

final_chian = classifier_chain | branch_chain

print(final_chian.invoke({'feedback' : 'this is a very bad rasgulla absulute a dogshit '}))