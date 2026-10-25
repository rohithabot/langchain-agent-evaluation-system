from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel

# Step 1: Configure local Ollama model
model = OllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

# Step 2: Create evaluation metric
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=model,
    include_reason=True
)

# Step 3: Define test case
test_case = LLMTestCase(
    input="What is Agentic AI?",
    actual_output=(
        "Agentic AI is a type of artificial intelligence "
        "that can make decisions, plan tasks, and execute "
        "actions autonomously using tools."
    )
)

# Step 4: Run evaluation
evaluate(
    test_cases=[test_case],
    metrics=[relevancy_metric]
)