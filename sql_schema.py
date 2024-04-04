import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()

# Initialize the SQLCoder model
sql_coder = LLM(
    model="TheBloke/sqlcoder-7B-AWQ",
    quantization="awq",
    dtype="float16",
    max_model_len=512,
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

class SQLRequest(BaseModel):
    schema: str
    text: str

class SQLResponse(BaseModel):
    sql_query: str

@app.post("/convert_text_to_sql", response_model=SQLResponse)
async def convert_text_to_sql(request: SQLRequest):
    sampling_params = SamplingParams(temperature=0.1, top_p=0.8, max_tokens=512)
    prompt_with_template = prompt_template.format(schema=request.schema, prompt=request.text)
    outputs = sql_coder.generate([prompt_with_template], sampling_params)
    sql_query = outputs[0].outputs[0].text if outputs else ""
    return SQLResponse(sql_query=sql_query)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
