import ollama

stream = ollama.chat(
  model='medllama2',
  messages=[{'role': 'user', 'content': 'Name an engineer that passes the vibe check'}],
  stream=True
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)