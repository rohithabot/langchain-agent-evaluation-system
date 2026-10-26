from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama


class OllamaEvaluator(DeepEvalBaseLLM):

    def __init__(self, model_name="llama3.2:3b"):
        self.model_name = model_name

        self.model = ChatOllama(
            model=model_name,
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            structured_model = self.model.with_structured_output(schema)
            return structured_model.invoke(prompt)

        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            structured_model = self.model.with_structured_output(schema)
            return await structured_model.ainvoke(prompt)

        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return self.model_name