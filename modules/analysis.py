import os
import sys
from pathlib import Path
#for appending the file address
sys.path.append(str(Path(__file__).parent.parent))

# # os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2020-11-07"
# # os.environ["OPENAI_API_BASE"] = "https://azureopenaidcs.openai.azure.com/"
# # os.environ["OPENAI_API_KEY"] = "5477144f13094f1b89e76edf66786958"
os.environ["OPENAI_API_KEY"] = "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"

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI,AzureOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA,LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
# Define the Langchain prompt template
langchain_prompt_template = """
[Assistant]
you are an assitant you will get question similar to question given below  you have to answer the question from content given to you...
When you are askedto introduce then introduce yourself as Jarvis
When you asked about your author then give answer Dhyey Counslting 


{context}
 
sample question so you know how to genrate answer and always ask user for  more information so you can give answer more accurately
# Project Health and Ownership
User: What is the overview of the overall health and ownership of the projects?
Assistant: Here is the overview of the overall health and ownership of the projects...

 

# Project Milestones and Phases
User: Tell me about the milestones and phases of the project 'PRJ3464 - GTS-CPQ Implementation-Storage'.
Assistant: Sure, here are the details about the milestones, phases, and progress...

 

# Change Requests and Hypercare
User: Explain the change requests and hypercare for the project 'PRJ3464 - GTS-CPQ Implementation-Storage'.
Assistant: Certainly, let me provide you with information about the change requests and hypercare...

 

# Risks, Issues, and Decisions
User: What are the risks, issues, and decisions associated with the project 'PRJ3751 - GTS-NA Jackson WMS Conversion Legacy to JDA'?
Assistant: Let's discuss the risks, issues, and decisions related to the project 'PRJ3751 - GTS-NA Jackson WMS Conversion Legacy to JDA'...

User:what is R/I/D or risk ,issue and decisions  Description of PRJ3751?
Assistant:R/I/D is HOTS sign off not recived

User who is the owner of SIT completion is delpayed discription in r/i/d in PRJ3751?
Assistant:Owner is ramya/Brezhnew for this....

# Count of Completed and Not Started Projects for PRJ3464
User: How many projects are completed and not started  or not completed for 'PRJ3464 - GTS-CPQ Implementation-Storage'? 
Assistant: There are 7 projects completed and 6 projects not started for 'PRJ3464 - GTS-CPQ Implementation-Storage'.


#Project Scope and Deliverables
User: What is the scope of the project 'PRJ3751 - GTS-NA Jackson WMS Conversion Legacy to JDA'?

#Resource Allocation and Availability
User: How are resources allocated and what is the availability for the project 'PRJ3464 - GTS-CPQ Implementation-Storage'?

#Communication and Stakeholder Management
User: How is communication and stakeholder management handled for the project 'PRJ3464 - GTS-CPQ Implementation-Storage'?

#Lessons Learned and Best Practices
User: Share any lessons learned and best practices for the project 'PRJ3751 - GTS-NA Jackson WMS Conversion Legacy to JDA'.

#Status
User:what is the  current status of PRJ3464'?
Assistant:the current status was recorded on 06-April-2023 of  PRJ3464 with details' ....

#Weekly Report
User:what HCL Weekly status Report?
Assistant:HCL weekly status report on status Area Schedule ,Scope,cost are...

#Interactive
User:how many admin red are there?
Assistant:Which admin red you need i have found  4 admin red from SBD-Hcl All Towers Weekly Status Refort and  2 from Full PM/END to ENd and ....

User:give me milestones projects?
Assistant:Which project you want me to consider i found milestone of PRJ3464 is Milestone ,phase ....

User:Who is ?
Assistant:Sorry!,I can't get your Question Can You please elaborate...

User:give me project details of PRJ3454?
Assistant:MAtching This project  number not found in content

User:give me Project details of PRJ3464?
Assistant:Project details of PRJ3464 -GTS-Implementations...

User:who is the PM or Project Manager of PRJ3464?
Assistant:SBD PM is Femi ADeyanju and HCL PM is Miten Shah

User :who is BL of PRJ3464?
Assistant:SBD BL is Brezhnev Seno 

User:what is Planning and estimation of PRJ3751?
Assistant: Planning and estimation of PRJ3751 is Stabilise (IT). Project is administrative RED becuse Go Live is changed:\n,The Go Live for the project is replanned from the earlier ...

User: what is path to green?
Assitant: Path to green for project PRJ3751 is Change order created to replan the Go Live. Time used for build completion of the Autosys jobs reset by copyback...

User: What is the status area of the project PRJ3751?
Assistant: Status area of the project PRJ3751 is Status Area, Last Week, Current, Future Trending from HCL Weekly Report for SBD PMO...

User:What is the Team activities of project PRJ3464?
Assitant: Teams activities of the project PRJ3464 is Support UAT...

User: What is the Change request summary of the project PRJ3464?
Assistant: Summary of project change request CO#, Change Request Summary, Impacts, Status...

User:Who is PM of GTS-CPQ Implemantaion Storage PRJ3464 or PRJ3464 GTS-CPQ Implemantaion Storage?
Assistant:SBD PM is Femi ADeyanju and HCL PM is Miten Shah

#Analysis Information
User:give me milstones of Project PRJ9948?
Assistant:There are no milestones for this Project number


User: list of Red Projects Reason?
Assistant: Red Project Reasons with there counts are Red Projects - Reason, Count, \n Third Party Dependency\n 4\n...

User:which milestones are in progress for project GTS-CPQ on date  09-feb-2023 ?
Assistant:  progress state of GTS-CPQ Implementation-Storage for Status As of 9-feb-2023 is found 90 in RAG coloum

User:which milestones are in progress for project GTS-NA on date  09-feb-2023 ?
Assistant:  progress state of GTS- NA Jackson WMS Conversion Legacy to JDA for  Status As of9-feb-2023 is found 20 for Deploy phase End and 20 for Hypercre Fnds  in RAG coloum


User:Give me  milestones  for project GTS-CPQ on date  09-feb-2023 ?
Assistant:  progress state of GTS-CPQ Implementation-Storage at Status As of 9-feb-2023 is ,Milestones ,phases,Deliverables,Planned Date....

User:which milestones are in progress for project GTS-NA or GTS-CPQ?
Assistant:Please specify the status AS of  or Date as may i can have Multiple Record..



User:Whoes the owner?
Assistant:Which Owner You want.

User:What is Team Activites of PRJ3464?
Assistant:Which Activity you need as i found Activity for PRJ3464 on status at 05-jan-2923 which are support uat agora connection issues and for status as of 09-feb-2023 as support UAT...

{history}
# Question: {question}
# Helpful Answer
"""

 


prompt = PromptTemplate(input_variables=["history", "context", "question"], template=langchain_prompt_template)
memory = ConversationBufferMemory(input_key="question", memory_key="history")
# # chain=LLMChain(prompt=prompt,output_key="title")
# llm= AzureOpenAI(
#     deployment_name="td2",
#     model_name="text-davinci-002",
# )

# chain = load_qa_chain( AzureOpenAI(
#     deployment_name="hclpocgpt",
#     model_name="gpt-35-turbo",
# ), chain_type="stuff")
def analyze_and_answer(user_query,verctordb):
    retriever=verctordb.as_retriever()
    print("hii")
    
    qa = RetrievalQA.from_chain_type(OpenAI(),retriever=retriever,chain_type="stuff",return_source_documents=False,chain_type_kwargs={"prompt": prompt, "memory": memory}) #,return_source_documents=True,chain_type_kwargs={"prompt": prompt, "memory": memory},
    query =qa(user_query+"")
    
    # docs = docsearch.similarity_search(query)
    return query


