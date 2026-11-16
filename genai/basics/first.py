from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st 
from dotenv import load_dotenv
load_dotenv()

client = ChatGroq(model= 'openai/gpt-oss-120b')
template = PromptTemplate(
    template='''

    You are an expert communication assistant. Write an email based on the following details:
    Purpose: {purpose}
    Tone: {tone}
    Recipient: {recipient}
    Name : {name}
    Recipient name : {recipient_name}
    
    Keep it clear, concise, and ready to send. Do not add placeholders like [contact],[date],e-t-c'.
'''
)

st.header('email generator ')
name = st.text_input('enter sender name ')
recipient_name = st.text_input('enter reciever name ')
perpose = st.selectbox('select perpose', [
    "Leave Request",
    "Job / Internship Application",
    "Client / Teacher Follow-up",
    "Apology / Delay Notice",
    "Networking / Cold Outreach"
])
tone = st.selectbox('selct tone ',[ 'Formal', 'Friendly', 'Urgent',  'Apologetic'])
recipient = st.selectbox('selct Recipient',['Boss', 'Teacher', 'Client', 'Friend'])

promt = template.invoke({
    'name':name,
    'recipient_name' : recipient_name,
    'purpose':perpose,
    'tone':tone,
    'recipient':recipient
})
output = client.invoke(promt)
if st.button('generate'):
    st.write(output.content)