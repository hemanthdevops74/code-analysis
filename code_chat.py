import streamlit as st
from dotenv import load_dotenv
import os
# import json
from google.auth import exceptions, default

from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.callbacks import StreamlitCallbackHandler
# import time


# from PyPDF2 import PdfReader
# from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
#from langchain.chat_models import ChatOpenAI


#-----
# Google Specific API's
from langchain.llms import VertexAI
# from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
# from langchain.memory import ConversationSummaryMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import MatchingEngine
#-----


credentials, _ = default()

# import os
# os.environ['GOOGLE_APPLICAITON_CREDENTIALS']='/Users/hemanthkumar74/Documents/langchain/chat_with_csv/credentials.json'
# print(os.environ['GOOGLE_APPLICAITON_CREDENTIALS'])

MODEL_NAME = "text-bison-32k"
MAX_OUTPUT_TOKENS="7000"
TEMPERATURE=0.0

CHUNK_SIZE=2000
CHUNK_OVERLAP=200

#Sample Python HelloChat app
#APP_PATH="/Users/hemanthkumar74/Documents/langchain/code_chat/sayhello"

# Sample Template
#DOC_TEMPLATE="/Users/hemanthkumar74/Documents/langchain/code_chat/Java_App_Details_Sample.docx"

#Spring Music
# APP_PATH = "/Users/hemanthkumar74/Documents/langchain/code_chat/spring-music"

# Spring MVC Sample
#APP_PATH="/Users/hemanthkumar74/Documents/langchain/code_chat/springmvc-hibernate-template"

# RabbitMQ Sample
# APP_PATH="/Users/hemanthkumar74/Documents/langchain/code_chat/rabbitmq-cloudfoundry-samples"

# Dot net
# APP_PATH="/Users/hemanthkumar74/Documents/langchain/code_chat/cf-dotnet-samples/dotnetcore"

#Pythons Sample from google.cloud
#APP_PATH="/Users/hemanthkumar74/Documents/langchain/code_chat/transactional-microservice-examples"

# Create a function to download the PDF
def download_pdf(llm_response, file_name):

  # Create a new PDF file.
  with open(file_name, "wb") as f:
    # Write the LLM response to the file.
    f.write(llm_response.content)

  # Close the file.
  f.close()

def main():
    # load_dotenv()
    st.set_page_config(page_title="Chat with Python Code base",
                       page_icon=":books:")

  
    text_area_default_value = """
1. Describe functionality about the application

2. List application dependencies along with versions in tabular format. Suggest latest version upgrades to these dependencies where possible

3. What are the potential security risks of using this code? Be specific and provide code remediations to the security risks

4. Create docker file to containerise this application and subsequently write gcloud scripts to deploy the image to Google Cloud Run

5. Suggest changes needed to the code to repoint the logs to Google Cloud Logging

6. Does the application use any Authentication/Authorization framwework? If yes, suggest changes needed to use Google Cloud IAM

7. 
    a. Generate a Jenkinsfile CI pipeline  for this application. The pipeline should also include executing unit test cases, static code check using the SonarQube tool
    b.  Generate a jenkinsfile deployment script CD pipeline to deploy to GKE from Google Artefact 
"""
    
    st.header("Chat with Python/Java Code base :books:")
    application_path = st.text_input("Enter the path of the application source code")
    front_end = st.selectbox("Select the Tech Stack",('Java','Python'))
    user_question = st.text_area("Ask a question about your code:",
                                 value=text_area_default_value)
    
    
    
    # if st.button("Start Analysis..."):
    #     if application_path and front_end and user_question:)
    if st.button('Start Analysis...'):
        st.write(front_end)
        if application_path and front_end and user_question and os.path.exists(application_path):
            APP_PATH = application_path
            global text_splitter
            if front_end == "Java" and APP_PATH:
                
                loader = GenericLoader.from_filesystem(
                    APP_PATH,
                    glob="**/*",
                    suffixes=[".java",".yaml"],
                    parser=LanguageParser(language=Language.JAVA, parser_threshold=500)
                    )
                documents = loader.load()   
                text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.JAVA, 
                                                                    chunk_size=CHUNK_SIZE, 
                                                                    chunk_overlap=CHUNK_OVERLAP)
            elif front_end == "Python" and APP_PATH:
            

                loader = GenericLoader.from_filesystem(
                    APP_PATH,
                    glob="**/*",
                    suffixes=[".py",".txt",".yaml"],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
                    )
                documents = loader.load()   
                text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                    chunk_size=CHUNK_SIZE, 
                                                                    chunk_overlap=CHUNK_OVERLAP)


        
            with st.spinner("Processing"):
                
                texts = text_splitter.split_documents(documents)
                # print(texts)
                # Perform a Maximal Marginal Relevance (MMR) search
                db = Chroma.from_documents(texts, VertexAIEmbeddings(disallowed_special=()))
                retriever = db.as_retriever(
                    search_type="mmr", # Also test "similarity"
                    search_kwargs={"k": 8}
                )
                
                # st_callback = StreamlitCallbackHandler(st.container())
                # message_holder = st.empty()
                # full_response=""
                llm = VertexAI(model_name=MODEL_NAME,
                                max_output_tokens=MAX_OUTPUT_TOKENS,
                                streaming=True,
                                temperature=TEMPERATURE,
                                verbose=True)
                # memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
                # qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
                # st.write(qa(user_question)['answer'])
                qa=RetrievalQA.from_chain_type(llm=llm,retriever=retriever,chain_type="stuff")
                prompt_template ="""You are an Application Architect analysing applications to modernize and migrate the application 
                to Google Cloud. You are looking to get as much technical information as possible from your application by scanning
                the code base. Think step by step.
                Provide your response accordingly to the user query given below.
                
                {query}
                """
                final_prompt= str(prompt_template.format(query=user_question))
                # print(final_prompt)
                # final_prompt = str(prompt) +"\n"+ str(user_question) 
                # print(qa.run(final_prompt))
                llm_response = qa.run(final_prompt)
                st.markdown(llm_response)
                # st.write(assistant_response)
                
                
                # code for printing in streaming format
                # for chunk in assistant_response.split():
                #         full_response += chunk + " "
                #         time.sleep(0.05)
                #         message_holder.markdown(full_response+" ")
                # message_holder.info(full_response)

                 # Create a button for the user to click to download the PDF
                download_button=st.download_button("Download analysis as HTML",data=llm_response,file_name="chat_response.html",mime="txt/html")
            
                

if __name__ == '__main__':
    main()
