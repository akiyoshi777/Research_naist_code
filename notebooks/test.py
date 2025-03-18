from ollama import Client

client = Client(host='http://benihi.naist.jp:11434/')
response = client.chat(model='command-r', messages=[
  {
    'role': 'user',
    'content': '糖尿病とは？',
  },
])
print(response['message']['content'])