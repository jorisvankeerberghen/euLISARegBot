import html
from mistralai import Mistral
import numpy as np
import faiss
import pickle
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import time
import glob

load_dotenv()

client = None
index = None
chunks = None

def init_euLISARegBot():
    global client, index, chunks

    # Check if the data is already initialized and stored
    # Define the model directory
    model_dir = 'model'

    # Define the paths for the stored files
    chunks_data_path = os.path.join(model_dir, 'chatbot_data.pkl')
    faiss_index_path = os.path.join(model_dir, 'faiss_index.bin')

    if os.path.exists(chunks_data_path) and os.path.exists(faiss_index_path):
        print("Loading chatbot data model...")
        with open(chunks_data_path,'rb') as f:
            chunks = pickle.load(f)

        index = faiss.read_index(faiss_index_path)

    else:
        print("Initializing eu-LISA Regulation ChatBot...")
        #Get Data from all HTML files in the source folder
        print("Loading EU regulations data...")
        source_folder = 'Regulations/'
        html_files = glob.glob(f"{source_folder}/*.html")
        
        articles = []
        
        for file in html_files:
            articles.extend(extract_articles_from_html(file))

        #Initialize the MistralClient
        print("Initializing the MistralClient...")
        key = os.getenv('MISTRAL_API_KEY')
        if not key:
            raise ValueError("MISTRAL_API_KEY environmnet variable is not set")
        client = Mistral(api_key=key)

        #Create embeddings for each text chunk
        print("Creating embeddings for each article...")
        text_embeddings = []
        chunks = articles
        
        selected_chunks = chunks #chunks [:5] + chunks[96:102]  Adjust this range to just test small parts 

        for i, chunk in enumerate(selected_chunks):  
            print(f"Processing chunk {i+1}/{len(chunks)}")
            # print(f"Chunk content:\n{chunk}\n")  # Debug Print the content of the chunk
            embedding = get_text_embedding(chunk)
            text_embeddings.append(embedding)
            
            #Introduce a delay to respect the 1 request per second rate limit from Mistral
            time.sleep(2)
         
        text_embeddings = np.array(text_embeddings)

        # Load into a vector database
        print("Loading embeddings into a vector database...")
        d = text_embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(text_embeddings)

        # Save the initialized data to a file
        # Ensure the 'model' directory exists
        model_dir = 'model'
        os.makedirs(model_dir, exist_ok=True)
        # Save the FAISS index to a file in the 'model' folder
        faiss_index_path = os.path.join(model_dir, 'faiss_index.bin')
        faiss.write_index(index, faiss_index_path)
        # Save the chunks data to a file in the 'model' folder
        chunks_data_path = os.path.join(model_dir, 'chatbot_data.pkl')
        with open(chunks_data_path, 'wb') as f:
            pickle.dump(chunks, f)
        

    print("Initializing the MistralClient...")
    if client is None:
        key = os.getenv('MISTRAL_API_KEY')
        if not key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        client = Mistral(api_key=key)
    print("Initialization complete")

def extract_articles_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        
    # Extract the regulation name from the HTML
    regulation_title_tag = soup.find("div", {"class": "eli-main-title"})
    if regulation_title_tag:
        regulation_title = regulation_title_tag.get_text(separator=" ", strip=True)
    else:
        regulation_title = "Unknown Regulation"
        
    articles = []
    
    #Count the number of articles in the legal basis
    article_count = 0
    while True:
        article_tag = soup.find("div", {"class": f"eli-subdivision", "id": f"art_{article_count + 1}"})
        if article_tag:
            article_count += 1
        else:
            break
    
    print(f"Total number of articles found: {article_count} in {regulation_title}")
    
    #Extract each article
    for i in range(1,article_count+1):
       article_tag = soup.find("div", {"class": f"eli-subdivision", "id": f"art_{i}"})
       if article_tag:
            article_text = article_tag.get_text(separator="\n", strip=True)
            articles.append({
                "text":article_text,
                "regulation":regulation_title,
                "article_number": i
                })
       
    return articles


def get_text_embedding(input):
    # Ensure that input is a string
    if isinstance(input,dict):
        input = input['text']
    
    embeddings_batch_response = client.embeddings.create(
        model="mistral-embed",
        inputs=input 
    )
    return embeddings_batch_response.data[0].embedding

def run_mistral(user_message, model):
    messages = [
        {
            "role":"user", "content": user_message
            }
    ]
    
    max_retries = 5
    delay = 1 #Wait time in seconds before retrying
    
    for attempt in range (max_retries):
        try:
            chat_response = client.chat.complete(
                model=model,
                messages=messages
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            if "Status 429" in str(e):
                print(f"Rate limit exceeded. Retrying in {delay} second...")
                time.sleep(delay)
            else:
                print(f"An error occured: {e}")
                break #break if a different exception
    else:
        print("Failed after 5 attempts due to rate limite issues.")
        return None

def chatbot_response(question,distance_threshold=0.5, max_chunks=20):
    #Create embeddings for the user question
    question_embeddings = np.array([get_text_embedding(question)])

    #Retrieve similar chunks from the vector database
    D, I = index.search(question_embeddings, k=max_chunks) #limit number of chunks if possible
    
    retrieved_chunks = []
    for distance, idx in zip(D[0], I[0]):
        if distance <= distance_threshold:
            retrieved_chunks.append(chunks[idx])
        else:
            break  # Stop if the similarity drops below the threshold

    # If no chunks meet the threshold, handle the case (e.g., lower the threshold or inform the user)
    if not retrieved_chunks:
        return "No relevant articles found for your query."

    context =""
    for chunk in retrieved_chunks:
        context += f"Regulation: {chunk['regulation']}, Article {chunk['article_number']}\n"
        context += chunk['text'] + "\n\n"
        
    #Combine context and question in a prompt and generate response
    model ="mistral-medium-latest"
    prompt = f"""
    You are provided with the following context, which includes specific articles from various regulations.
    Please answer the following query by enumerating each relevant point, and referring to each and every regulation and article number from the context. When referring to the regulations, ensure the following format is used: 

    - Regulation (EU) 2019/817 (IO):
    - Regulation (EU) 2018/1240 (ETIAS):
    - Regulation (EU) 2018/1726 (eu-LISA):
    - Regulation (EU) 2017/2226 (EES):

    Structure your answer with clearly numbered points, each corresponding to a different regulation or article as applicable.

    Context:
    ---------------------------
    {context}
    Query: {question}
    Answer:
    1. 
    2. 
    3. 
    4. 
    """
    response = run_mistral(prompt, model)
    return response

