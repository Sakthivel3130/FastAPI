from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-1_8B-Chat-Int4",
    device_map="auto",  
    trust_remote_code=True
).eval()


@app.post("/chatbot/")
async def chatbot(msg: str):
    response, _ = model.chat(tokenizer, "My colleague works diligently", history=None, system=msg)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
