from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

model = ChatGroq(model='openai/gpt-oss-120b')
template1 = PromptTemplate.from_template(
    template='give me a detialed discription about {topic}'
)
template2 = PromptTemplate.from_template(
    template='give me a 5 line summary of  {text}'
)
# withoutchain
print ('-=-=-=-=--=-=-=-=-=-=-= with out chain outtput =-=-=-=-=-=-=-=-=-=-=-=-')
promt1 = template1.format(topic = 'black holes')
result = model.invoke(promt1)
promt2 = template2.format(text = result)
print((model.invoke(promt2)).content)


#  with chain 
print('-=-=-=-=-=-=-=-=-=-= with chain output =-=-=-=-=-=-=-=-=-=-=-=-')
parser = StrOutputParser()
chain = template1 | model | parser |template2 |model |parser

final_result = chain.invoke({'topic':'black holes'})

print (final_result)

print()
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= json out put parser =-=-=-=-=-=-=-=-=-=-=')

parser1 = JsonOutputParser()
template = PromptTemplate(template='give identical details of a fictional person \n {parser}',
                          input_variables=[],
                          partial_variables={'parser':parser1.get_config_jsonschema()})
promt = template.invoke(input={})
print(model.invoke(promt).content)

print()
print('-=-==-=-=-=-=-=-=-=-=- json parser with chain')

chain = template | model | parser 
print(chain.invoke({}))


print()
print('-=-=-=-=--0--=-=-=-=-=-=-=-=- with pydentic parser -=-=-=-=-=-=-=-=-=-=-=-=')
class Person (BaseModel):
    name : str = Field(description = 'name of the person')
    age : int = Field(description = 'age of the person',ge=18)
    city : str = Field (description = 'name of the city where the person lives ')

parser = PydanticOutputParser(pydentic_object = Person)

template3 = PromptTemplate(
    template='genrate name age city of a fictional {place} \n {parser}',
    input_variables=['place'],
    partial_variables={'parser':parser.get_format_instructions()}
    )

print('\n -=-=-=-=-= without chain -=-=-=-=-=-=-=')
promt = template3.invoke(input={'place':'pakistani'})
print(model.invoke(promt).content)

print('\n -=-=-=-=-=-=-= with chain -=-=-=-=-=-=-=-=')

chain = template3 | model  | parser
print(chain.invoke({'place':'pakistani'}))

