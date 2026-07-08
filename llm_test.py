import os

import openai

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


HARDCODED_API_KEY = "sk-8666943ae667410fb46edb686184c37c"

API_KEY = os.getenv("DASHSCOPE_API_KEY", HARDCODED_API_KEY)
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3.7-plus")


if not API_KEY:
    raise RuntimeError("请先在 HARDCODED_API_KEY 中填写 API Key，或设置环境变量 DASHSCOPE_API_KEY。")


messages = [
        {"role": "system", "content": "You are a helpful assistant for geographic information."},
        {"role": "user", "content": "你是谁？"},
]


if OpenAI is not None:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    print(completion.choices[0].message.content)
else:
    openai.api_key = API_KEY
    openai.api_base = BASE_URL
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
    )
    print(completion["choices"][0]["message"]["content"])
