from deepeval.models import OllamaModel

model = OllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

response = model.generate(
    "Explain Agentic AI in one sentence."
)

print("Ollama Response:")
print(response)