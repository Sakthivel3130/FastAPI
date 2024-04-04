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
This query will run on a database whose schema is represented in this string:
CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product\n
  name VARCHAR(50), -- Name of the product\n
  price DECIMAL(10,2), -- Price of each unit of the product\n
  quantity INTEGER  -- Current quantity in stock\n
);

CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale\n
  product_id INTEGER, -- ID of product sold\n
  customer_id INTEGER,  -- ID of customer who made purchase\n
  salesperson_id INTEGER, -- ID of salesperson who made the sale\n
  sale_date DATE, -- Date the sale occurred\n
  quantity INTEGER -- Quantity of product sold\n
);

-- sales.product_id can be joined with products.product_id

### SQL
Given the database schema, here is the SQL query that answers `{prompt}`:
```sql
'''


class SQLRequest(BaseModel):
    text: str

class SQLResponse(BaseModel):
    sql_query: str

@app.post("/convert_text_to_sql", response_model=SQLResponse)
async def convert_text_to_sql(request: SQLRequest):
    sampling_params = SamplingParams(temperature=0.1, top_p=0.8, max_tokens=512)
    prompt_with_template = prompt_template.format(prompt=request)
    outputs = sql_coder.generate([prompt_with_template], sampling_params)
    sql_query = outputs[0].outputs[0].text if outputs else ""
    return SQLResponse(sql_query=sql_query)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
