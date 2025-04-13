import os
import pymongo
from langchain_community.document_loaders import PyPDFLoader, UnstructuredExcelLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_community.docstore.document import Document
import requests.exceptions
import glob
from flask import Flask, request, jsonify
from geopy.distance import geodesic
import traceback
import re


app = Flask(__name__)

client = pymongo.MongoClient("mongodb+srv://projectDB:PEyHwQ2fF7e5saEf@cluster0.43hxo.mongodb.net/")
db = client["ta7t-bety"]

collection_1 = db["users"]
collection_2 = db["providers"]
collection_3 = db["knowledge_base"]
collection_4 = db["posts"]
 
TEXT_FILE_PATH = r"C:\Users\mostafa\Desktop\cahtbot document.txt"

def load_mongo_data():
    print("\nAttempting to load data from MongoDB collections...")
    
    try:
        docs = []
        
        for collection in [collection_1, collection_2, collection_3, collection_4]:
            for doc in collection.find():
                
                content = f"Document ID: {doc['_id']}\n"
                
                if collection.name == "providers" and "locations" in doc:
                    for key, value in doc.items():
                        if key == "locations":
                            content += f"{key}:\n"
                            for loc in value:
                                if "coordinates" in loc and isinstance(loc["coordinates"], list) and len(loc["coordinates"]) >= 2:
                                    content += f"  - Address: {loc.get('address', 'Unknown')}\n"
                                    content += f"  - Longitude: {loc['coordinates'][0]}\n"
                                    content += f"  - Latitude: {loc['coordinates'][1]}\n"
                                else:
                                    content += f"  - {loc}\n"
                        else:
                            content += f"{key}: {value}\n"
                else:
                    for key, value in doc.items():
                        if key != "_id":
                            content += f"{key}: {value}\n"
                
                doc_obj = Document(
                    page_content=content,
                    metadata={"source": collection.name, "doc_id": str(doc["_id"])}
                )
                docs.append(doc_obj)
        
        print(f"Successfully created {len(docs)} documents from MongoDB collections")
        return docs
    
    except Exception as e:
        print(f"Error processing MongoDB data: {str(e)}")
        return []

def load_specific_text_file():
    print(f"\nAttempting to load text file from: {TEXT_FILE_PATH}")
    
    docs = []
    
    try:
        if not os.path.exists(TEXT_FILE_PATH):
            print(f"Warning: Text file not found at {TEXT_FILE_PATH}")
            return []
        
        try:
            loader = TextLoader(TEXT_FILE_PATH)
            file_documents = loader.load()
            
            for doc in file_documents:
                doc.metadata["source"] = f"text_file:chatbot_document"
            
            docs.extend(file_documents)
            print(f"Successfully loaded text file: {TEXT_FILE_PATH}")
        except Exception as e:
            print(f"Error loading file {TEXT_FILE_PATH}: {str(e)}")
        
        return docs
    
    except Exception as e:
        print(f"Error processing text file: {str(e)}")
        return []

embeddings = OllamaEmbeddings(model="all-minilm")

def initialize_system():
    global vectorstore, retriever, rag_chain, conversational_chain

    mongo_documents = load_mongo_data()
    text_documents = load_specific_text_file()

    all_documents = mongo_documents + text_documents

    if not all_documents:
        print("No documents were loaded. Please check if the MongoDB connection is successful and the text file exists.")
        return False

    print(f"\nTotal documents loaded: {len(all_documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(all_documents)
    print(f"\nCreated {len(splits)} text chunks")

    print("Creating vector store...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )

    vectorstore.save_local("faiss_index")
    print("Vector store created and saved")

    llm = OllamaLLM(model="llama3.2")

    rag_template = """Answer the following question based on the provided context.
    The answer should be concise and direct, focusing only on relevant information.

    Follow these specific rules:
    1. For user questions:
       - Display the name, position, region, department, age, and location in a concise way.
    2. For service provider questions:
       - Provide provider details with name, services, and pricing briefly.
       - Clearly include longitude and latitude values.
    3. For knowledge base questions:
       - Provide only the most relevant content to the topic.
    4. For post questions:
       - Provide only the post title, content, and price in an organized format.
    5. For text file questions:
       - Share only the relevant content from the chatbot document.

    Context: {context}

    Question: {question}

    Answer using information from the context only. If you are unable to answer this question based on the available information,
    say "I cannot answer this question based on the available information."

    Provide a concise, clear, and easy-to-read answer:"""

    rag_prompt = PromptTemplate(
        template=rag_template,
        input_variables=["context", "question"]
    )

    conversational_template = """You are a helpful assistant for a system called ta7t-bety.

    Question: {question}

    Respond in a friendly and concise manner in English. If the question seems like a greeting or simple conversation, respond appropriately and briefly.
    
    Provide a direct and concise answer:"""

    conversational_prompt = PromptTemplate(
        template=conversational_template,
        input_variables=["question"]
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt 
        | llm 
        | StrOutputParser()
    )

    conversational_chain = (
        {"question": RunnablePassthrough()}
        | conversational_prompt
        | llm
        | StrOutputParser()
    )
    
    return True

def format_response(response):
    response = response.strip()
    
    if any(ord(char) >= 0x0600 and ord(char) <= 0x06FF for char in response):
        english_sections = re.findall(r'\(.*?\)', response)
        for section in english_sections:
            if all(ord(char) < 0x0600 or ord(char) > 0x06FF or char in ' .,!?():-' for char in section):
                response = response.replace(section, '')
    
    response = re.sub(r'First,|Second,|Third,', '', response)
    response = re.sub(r'\n{3,}', '\n\n', response)
    response = re.sub(r'Located at coordinates: \[([\d\.-]+), ([\d\.-]+)\]', 
                     r'Location: Latitude \1, Longitude \2', response)
    response = re.sub(r'Let me think|Based on the context|According to the provided information', '', response)
    response = re.sub(r' {2,}', ' ', response)
    response = re.sub(r'([.,:;?!])([^ \n])', r'\1 \2', response)
    
    return response
 
def is_conversational_query(question):
    conversational_patterns = [
        "hi", "hello", "hey", "how are you", "what's up", "good morning", "good afternoon",
        "good evening", "nice to meet you", "how's it going", "what can you do",
        "who are you", "help", "thanks", "thank you", "goodbye", "bye",
        "hello", "greetings", "hi", "how are you", "what's up", "good morning", 
        "good evening", "thank you", "thanks", "goodbye"
    ]
    
    question_lower = question.lower().strip()
    for pattern in conversational_patterns:
        if pattern in question_lower or question_lower == pattern:
            return True
    
    if len(question_lower.split()) <= 3:
        return True
    
    return False

def query_documents(question: str):
    try: 
        if is_conversational_query(question):
            response = conversational_chain.invoke(question)
        else:
            response = rag_chain.invoke(question)
            
        return format_response(response)
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama service. Please make sure the service is running."
    except Exception as e:
        return f"Error: {str(e)}"

def refresh_data():
    print("\nRefreshing data from all sources...")
    
    global vectorstore, retriever
    
    mongo_docs = load_mongo_data()
    text_docs = load_specific_text_file()
    
    all_docs = mongo_docs + text_docs
    
    if not all_docs:
        print("Warning: No documents were loaded during refresh.")
        return False
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    splits = text_splitter.split_documents(all_docs)
    print(f"\nCreated {len(splits)} text chunks")
    
    print("Creating updated vector store...")
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
     
    vectorstore.save_local("faiss_index")
    print("Vector store updated and saved")
     
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    return True
 
print("Initializing RAG system...")
initialize_system()
 
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Question field is required'
            }), 400
            
        question = data['question']
        if not question.strip():
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400
        
        response = query_documents(question)
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': response
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    try:
        success = refresh_data()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Data updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to update data'
            }), 500
            
    except Exception as e: 
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    

@app.route('/api/find_restaurants', methods=['POST'])
def find_nearby_restaurants():
    try:
        data = request.get_json()

        if not data or 'latitude' not in data or 'longitude' not in data:
            return jsonify({
                'success': False,
                'error': 'Required fields missing: latitude, longitude'
            }), 400

        user_latitude = float(data['latitude'])
        user_longitude = float(data['longitude'])
         
        if abs(user_latitude) > 90:  
            user_latitude, user_longitude = user_longitude, user_latitude
            
        user_location = (user_latitude, user_longitude)
        
        print(f"User location (corrected): {user_location}")
        
        search_radius = float(data.get('radius', 25))
        
        nearby_restaurants = []
        
        all_providers = list(collection_2.find({}))
        
        print(f"Total documents in collection: {len(all_providers)}")
        
        for provider in all_providers:
            provider_name = provider.get("name", "Unknown restaurant")
            provider_id = provider.get("_id", "Unknown ID")
            
            if "locations" in provider and isinstance(provider["locations"], list):
                for loc in provider["locations"]:
                    if "coordinates" in loc and isinstance(loc["coordinates"], list) and len(loc["coordinates"]) >= 2:
                        raw_coords = loc["coordinates"]
                        
                        coord1, coord2 = raw_coords[0], raw_coords[1]
                        
                        if abs(coord1) <= 90 and abs(coord2) <= 180:
                            latitude, longitude = coord1, coord2
                        elif abs(coord2) <= 90 and abs(coord1) <= 180:
                            latitude, longitude = coord2, coord1
                        else:
                            latitude, longitude = coord2, coord1
                        
                        provider_location = (latitude, longitude)
                        
                        try:
                            distance = geodesic(user_location, provider_location).kilometers
                            
                            if distance <= search_radius:
                                nearby_restaurants.append({
                                    "name": provider_name,
                                    "address": loc.get("address", "Unknown address"),
                                    "distance_km": round(distance, 2),
                                    "coordinates": {
                                        "latitude": latitude,
                                        "longitude": longitude
                                    }
                                })
                        except Exception as e:
                            print(f"Error calculating distance: {str(e)}")
        
        formatted_response = {
            'success': True,
            'message': f"Found {len(nearby_restaurants)} restaurants near you",
            'restaurants': sorted(nearby_restaurants, key=lambda x: x["distance_km"])
        }
        
        return jsonify(formatted_response)

    except Exception as e:
        print(f"Error in find_nearby_restaurants: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"An error occurred: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'message': 'RAG system is operational'
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)