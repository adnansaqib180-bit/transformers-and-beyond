from typing import TypedDict, Annotated, Optional, List,Literal
from dotenv import load_dotenv
from langchain_groq import ChatGroq


class Employee(TypedDict):
  name: Annotated[str, "name of the person applying"]
  experience: Annotated[int, "how much experience does the person applying have?"]
  skills: Annotated[List[str], "all the skills of the applicant"]
  available: Annotated[
      Optional[bool], "is the applicant available for work?"
  ]


load_dotenv()
client = ChatGroq(model="openai/gpt-oss-120b")
structured_model = client.with_structured_output(Employee,method='json_mode')

result = structured_model.invoke("""
        Extract the following details in JSON format (keys: name, experince, skills, available):
        Assalam-o-Alaikum! My name is Sara Khan and I am based in Lahore. 
        You can reach me at sara.khan99@email.com or call me on 0300-1234567. 
        I have a total of 5 years of experience in data science and machine learning. 
        I started my career at Systems Limited as a Junior Data Analyst where 
        I worked for 2 years. After that, I joined Careem as a Senior ML Engineer and 
        worked there for 3 years. My main tech stack includes Python, Pandas, PyTorch, 
        AWS, and Docker. Right now I am free and can join immediately if the offer is good. 
        I am looking for something around 350,000 PKR per month.
    """)
print(result)

# ---------------------------------------------------------------------------------------
# saame with pydentic ----------------------=------------

from pydantic import BaseModel, Field


class ContactInfo(BaseModel):
  email: str = Field(description="Candidate email")
  phone: str = Field(description="Candidate phone number")
  city: str = Field(description="Candidate city")


class WorkExperience(BaseModel):
  company_name: str = Field(description="Company name")
  role: str = Field(description="Job role")
  duration_years: int = Field(description="Years worked")


class CandidateProfile(BaseModel):
  candidate_name: str = Field(description="Full name")
  years_of_experience: int = Field(description="Total experience in years")
  skills: List[str] = Field(description="Technical skills")
  is_available_immediately: bool = Field(description="True if free to join now")
  contact_info: ContactInfo
  work_history: List[WorkExperience]
  expected_salary_pkr: int = Field(description="Expected salary in PKR")
  summary_tone: Literal["Confident", "Neutral", "Hesitant", "Formal"]


load_dotenv()
client = ChatGroq(model="openai/gpt-oss-120b")
structured_model = client.with_structured_output(
    CandidateProfile, method="json_mode"
)

text = """
Assalam-o-Alaikum! My name is Sara Khan and I am based in Lahore. 
You can reach me at sara.khan99@email.com or call me on 0300-1234567. 
I have a total of 5 years of experience in data science and machine learning. 
I started my career at Systems Limited as a Junior Data Analyst where 
I worked for 2 years. After that, I joined Careem as a Senior ML Engineer and 
worked there for 3 years. My main tech stack includes Python, Pandas, PyTorch, 
AWS, and Docker. Right now I am free and can join immediately if the offer is good. 
I am looking for something around 350,000 PKR per month.
"""

result = structured_model.invoke(
    f"Extract the data in JSON format strictly following this schema:\n"
    f"{CandidateProfile.model_json_schema()}\n\nText:\n{text}"
)
import json
print(json.dumps(result.model_dump(), indent=2))
