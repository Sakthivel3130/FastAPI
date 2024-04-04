import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

app = FastAPI()

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,
    device_map="cuda:0"
)

# Using the text streamer to stream output one token at a time
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "Tell me about AI"
prompt_template=f'''<s>[INST] {prompt} [/INST]
'''
generation_params = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1
}
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_params
)
class Message(BaseModel):
    text: str

@app.post("/chatbot/")
async def chatbot(message: Message):
    pipe_output = pipe(prompt_template)[0]['generated_text']
    print("pipeline output: ", pipe_output)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)


