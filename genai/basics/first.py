from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st 

client = ChatGroq(model= 'openai/gpt-oss-120b')
template = PromptTemplate(
    template='''

    You are an expert communication assistant. Write an email based on the following details:
    Purpose: {purpose}
    Tone: {tone}
    Recipient: {recipient}
    
    Keep it clear, concise, and ready to send. Do not add placeholders like [Your Name], just end with 'Best regards'.
'''
)

st.header('email generator ')
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
    'purpose':perpose,
    'tone':tone,
    'recipient':recipient
})
output = client.invoke(promt)
if st.button('generate'):
    st.write(output.content)