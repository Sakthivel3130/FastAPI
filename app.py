from fastapi import FastAPI,UploadFile,File
from faster_whisper import WhisperModel
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from transformers import pipeline
import uvicorn
import librosa
import torch
import gc


app = FastAPI()

class Message(BaseModel):
    text: str

class Translations(BaseModel):
    text:str
    target_lang:str

class SQLResponse(BaseModel):
    sql_query: str

class SQLRequest(BaseModel):
    schema: str
    text: str


@app.post("/convert_text_to_sql")
async def convert_text_to_sql(request: SQLRequest):
    #model_name =
    sql_coder = LLM(
                model= "TheBloke/sqlcoder-7B-AWQ",
                quantization="awq",
                dtype="half",
                max_model_len=1000
            )
    prompt_template = '''## Task
    Generate a SQL query to answer the following question:
    `{prompt}`

    ### Database Schema
    {schema}

    ### SQL
    Given the database schema, here is the SQL query that answers `{prompt}`:
    ```sql
    '''
    sampling_params = SamplingParams(temperature=0, max_tokens=300)
    prompt_with_template = prompt_template.format(schema=request.schema,prompt=request.text)
    outputs = sql_coder.generate([prompt_with_template], sampling_params)
    sql_query = outputs[0].outputs[0].text if outputs else ""
    destroy_model_parallel()
    del sql_coder
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    return sql_query

@app.post("/mistral_chatbot")
async def chatbot(message: Message):
    prompt_template =f'''<s>[INST]{{prompt}}[/INST]\n'''
    mistral_llm = LLM(
                model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
                quantization="awq",
                dtype="half",
                max_model_len=1000
            )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)
    prompt_with_template = prompt_template.format(prompt=message)
    response = mistral_llm.generate([prompt_with_template], sampling_params)
    destroy_model_parallel()
    del mistral_llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    return response[0].outputs[0].text


@app.post("/Transcription")
async def Audio_text(file:UploadFile=File(...)):
    model = WhisperModel("large-v2")
    seg = " "
    audio_content,sr = librosa.load(file.file)
    segments, info = model.transcribe(audio_content)
    for segment in segments:
        seg += "%s " % segment.text
    del model
    torch.cuda.empty_cache()
    return seg

@app.post("/llama_chatbot")
async def chatbot(message: Message):
    prompt_template='''[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
    <</SYS>>
    {prompt}[/INST]'''

    llm = LLM(
        model="TheBloke/Llama-2-7B-Chat-AWQ",
        quantization="awq",
        dtype="half",
        max_model_len=1000
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)
    prompt_with_template = prompt_template.format(prompt=message)
    response = llm.generate([prompt_with_template], sampling_params)
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    return response[0].outputs[0].text

@app.post("/Translation")
async def Text_Translation(request: Translations):
    model_name = "facebook/m2m100_418M"
    pipe = pipeline("text2text-generation", model=model_name)
    text = request.text
    target_lang = request.target_lang
    translated_text = pipe(text, forced_bos_token_id=pipe.tokenizer.get_lang_id(lang=target_lang))
    generated_text = translated_text[0]['generated_text']
    del pipe
    torch.cuda.empty_cache()
    return generated_text

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
