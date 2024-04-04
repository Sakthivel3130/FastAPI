import requests

url= "http://127.0.0.1:8001"

audio_file_path = "/home/team1/Sakthivel_f22/fastapi/sample2.wav" 

transcription_endpoint = f"{url}/Transcription"
translation_endpoint = f"{url}/Translation"
text_generation_endpoint = f"{url}/mistral_chatbot"
chatbot_endpoint = f"{url}/llama_chatbot"
sql_conversion_endpoint = f"{url}/convert_text_to_sql"


message_data = {"text": "What is AI?"}
translation_data = {"target_lang": "fr", "text": "Hello, how are you?"}
sql_conversion_data = {"schema":"products(product_id INTEGER PRIMARY KEY,product_name VARCHAR(50)price DECIMAL(10,2),quantity INTEGER);","text": "Show all products with a price greater than 100"}

def call_api(endpoint, data=None, files=None):
    url = endpoint
    if data:
        response = requests.post(url, json=data)
    elif files:
        with open(files, 'rb') as f:
            response = requests.post(url, files={'file': f})
    else:
        response = requests.post(url)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error {response.status_code}: {response.reason}"}


transcription_response = call_api(transcription_endpoint, files=audio_file_path)
print("Transcription:", transcription_response)

translation_response = call_api(translation_endpoint, data=translation_data)
print("Translation:", translation_response)

text_generation_response = call_api(text_generation_endpoint, data=message_data)
print("Text Generation:", text_generation_response)

chatbot_response = call_api(chatbot_endpoint, data=message_data)
print("Chatbot Response:", chatbot_response)

sql_conversion_response = call_api(sql_conversion_endpoint, data=sql_conversion_data)
print("SQL Conversion:", sql_conversion_response)

