import google.generativeai as genai
import pandas as pd 
import numpy as np 
import joblib
import os 
import json 
from sklearn.metrics.pairwise import cosine_similarity
import requests
genai.configure(api_key="AIzaSyCPfJ9bt2N7Jc1fsGycU7SohEH1lGGTyBU")
def create_embeddings(text_list):
    r=requests.post("http://localhost:11434/api/embed",
                json={
                    "model":"bge-m3",
                    "input": text_list
                } 
             
                )
    embedding = r.json()['embeddings']
    return embedding


model = genai.GenerativeModel("gemini-2.5-flash") 
def inference(prompt):
    response = model.generate_content(prompt)
    return response.text
# def inference(prompt):
#      r=requests.post("http://localhost:11434/api/generate",
#                 json={
#                     "model":"llama3.2",
#                     "prompt": prompt,
#                     "stream":False
#                 } 
             
#                 ) 
#      response = r.json()
#     #  print(response)
#      return response
 
df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question:")

Question_embeddings = create_embeddings(incoming_query)[0]
# print(Question_embeddings)

# findding the cosine similarity to finding the vest answer from this data 

similarities = cosine_similarity(np.vstack(df['embedding']),[Question_embeddings]).flatten()
# print(similarities)

top_results = 3
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)

new_df = df.loc[max_indx]
# print(new_df[['crop','soil_type','fertilizers','text']])




prompt = f'''
You are an agricultural assistant helping farmers in Uttar Pradesh. Based on the following query, retrieve the most relevant information about crop cultivation, soil type, temperature, diseases, and fertilizers. Respond in simple language suitable for rural farmers.
{new_df[['crop','soil_type','fertilizers',"diseases",'text']].to_json(orient = 'records')}
------------------------------------------------------------------

Query:"{incoming_query}" process this query and answer the question perfectly in brief if user aks in detail, give him/her in detailed answer strictly follow this line.Don't ask it 
Would you like to know more about how to prevent or manage these diseases?.You have to remmeber the previous questions and then give the answer of the asked question 
 
'''


with open ("prompt.txt","w") as f:
    f.write(prompt)

response = inference(prompt) #['response'] without gemini it have to used for project 

print(response)
    
with open ("response.txt","w") as f:
    f.write(response)
