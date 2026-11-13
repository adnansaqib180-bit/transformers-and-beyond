from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROCK"),
)

def final_promt():
    return '''please retrive the company name, address, 
    phone number, email, and website from the following text if not present
    then return "not present" in the respective field. for website if not 
    present then you can use external sources still can't find then return "not present".
    return the information in json fromat should be like this:
    {
    
        "company_name": "company name",
        "address": "address",
        "phone_number": "phone number",
        "email": "email",
        "website": "website"
    }
    
'''

def extract_information(text):
    prompt = final_promt() + text
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

while True:
    user_input = input("Enter text to extract information (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    extracted_info = extract_information(user_input)
    print("Extracted Information: " + extracted_info)