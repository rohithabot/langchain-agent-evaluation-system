from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel

# Step 1: Configure local Ollama model
model = OllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434"
)

# Step 2: Configure faithfulness metric
faithfulness_metric = FaithfulnessMetric(
    threshold=0.7,
    model=model,
    include_reason=True
)

# Step 3: Define knowledge base context
retrieval_context = [
    (
        "Agentic AI is a type of artificial intelligence "
        "that can make decisions, plan tasks, and execute "
        "actions using tools."
    ),
    (
        "AI agents can use tools such as calculators, "
        "search engines, databases, and document retrieval systems."
    )
]

# Step 4: Define evaluation test case
test_case = LLMTestCase(
    input="What is Agentic AI?",
    actual_output=(
        "Agentic AI is a type of artificial intelligence "
        "that can make decisions, plan tasks, and execute "
        "actions using tools such as calculators and search engines."
    ),
    retrieval_context=retrieval_context
)

# Step 5: Run evaluation
evaluation_result = evaluate(
    test_cases=[test_case],
    metrics=[faithfulness_metric]
)

# Step 6: Read results from evaluation output
print("\n========== FAITHFULNESS RESULTS ==========")

for test_result in evaluation_result.test_results:

    print("Test Case:", test_result.name)
    print("Test Passed:", test_result.success)

    for metric_result in test_result.metrics_data:

        print("\nMetric:", metric_result.name)
        print("Score:", metric_result.score)
        print("Threshold:", metric_result.threshold)
        print("Passed:", metric_result.success)
        print("Reason:", metric_result.reason)

print("==========================================")