import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
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

class Message(BaseModel):
    text: str

@app.post("/chatbot/")
async def chatbot(message: Message):
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)
    prompt_with_template = prompt_template.format(prompt=message)
    response = llm.generate([prompt_with_template], sampling_params)
    return {"response": response[0].outputs[0].text}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
