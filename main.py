from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

auth_token = "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"


tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=auth_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto", token=auth_token)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
