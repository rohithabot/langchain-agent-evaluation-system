import json
from pathlib import Path

from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

from ollama_evaluator import OllamaEvaluator


# ==========================================
# STEP 1: Load agent results
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

results_path = BASE_DIR / "agent_results.json"

with open(results_path, "r", encoding="utf-8") as file:
    agent_results = json.load(file)

print("Agent results loaded successfully!")
print("Total test cases:", len(agent_results))


# ==========================================
# STEP 2: Initialize local Ollama evaluator
# ==========================================

evaluator_model = OllamaEvaluator(
    model_name="llama3.2:3b"
)

print("Ollama evaluator initialized successfully!")


# ==========================================
# STEP 3: Convert to DeepEval test cases
# ==========================================

deepeval_test_cases = []

for result in agent_results:

    actual_tools = [
        ToolCall(name=tool["name"])
        for tool in result.get("tool_calls", [])
    ]

    expected_tools = [
        ToolCall(name=result["expected_tool"])
    ]

    test_case = LLMTestCase(
        input=result["question"],
        actual_output=result.get("answer", ""),
        tools_called=actual_tools,
        expected_tools=expected_tools,
        retrieval_context=result.get("retrieval_context") or None
    )

    deepeval_test_cases.append(test_case)


# ==========================================
# STEP 4: Run evaluations
# ==========================================

evaluation_results = []

for index, test_case in enumerate(
    deepeval_test_cases,
    start=1
):

    print("\n" + "=" * 60)
    print(f"TEST CASE {index}")
    print("=" * 60)

    print("Question:", test_case.input)

    result = agent_results[index - 1]

    # --------------------------------------
    # 4.1 Tool correctness
    # --------------------------------------

    expected_names = [
        tool.name for tool in test_case.expected_tools
    ]

    actual_names = [
        tool.name for tool in test_case.tools_called
    ]

    # Require exactly the expected tool sequence.
    tool_passed = expected_names == actual_names

    print("\nTOOL CORRECTNESS")
    print("Expected:", expected_names)
    print("Actual:", actual_names)
    print("Passed:", tool_passed)

    # --------------------------------------
    # 4.2 Answer relevancy
    # --------------------------------------

    relevancy_score = None
    relevancy_passed = None
    relevancy_reason = None

    if result.get("answer"):

        try:

            relevancy_metric = AnswerRelevancyMetric(
                threshold=0.7,
                model=evaluator_model,
                include_reason=True,
                async_mode=False
            )

            relevancy_metric.measure(test_case)

            relevancy_score = relevancy_metric.score
            relevancy_passed = relevancy_metric.is_successful()
            relevancy_reason = relevancy_metric.reason

            print("\nANSWER RELEVANCY")
            print("Score:", relevancy_score)
            print("Passed:", relevancy_passed)
            print("Reason:", relevancy_reason)

        except Exception as error:

            print("\nANSWER RELEVANCY ERROR:")
            print(str(error))

    # --------------------------------------
    # 4.3 Faithfulness
    # --------------------------------------

    faithfulness_score = None
    faithfulness_passed = None
    faithfulness_reason = None

    if test_case.retrieval_context:

        try:

            faithfulness_metric = FaithfulnessMetric(
                threshold=0.7,
                model=evaluator_model,
                include_reason=True,
                async_mode=False
            )

            faithfulness_metric.measure(test_case)

            faithfulness_score = faithfulness_metric.score
            faithfulness_passed = faithfulness_metric.is_successful()
            faithfulness_reason = faithfulness_metric.reason

            print("\nFAITHFULNESS")
            print("Score:", faithfulness_score)
            print("Passed:", faithfulness_passed)
            print("Reason:", faithfulness_reason)

        except Exception as error:

            print("\nFAITHFULNESS ERROR:")
            print(str(error))

    # --------------------------------------
    # Store evaluation results
    # --------------------------------------

    evaluation_results.append({
        "question": test_case.input,
        "tool_correctness": {
            "passed": tool_passed,
            "expected": expected_names,
            "actual": actual_names
        },
        "answer_relevancy": {
            "score": relevancy_score,
            "passed": relevancy_passed,
            "reason": relevancy_reason
        },
        "faithfulness": {
            "score": faithfulness_score,
            "passed": faithfulness_passed,
            "reason": faithfulness_reason
        }
    })


# ==========================================
# STEP 5: Save evaluation results
# ==========================================

output_path = BASE_DIR / "evaluation_results.json"

with open(output_path, "w", encoding="utf-8") as file:

    json.dump(
        evaluation_results,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nEvaluation results saved to:")
print(output_path)


# ==========================================
# STEP 6: Final summary
# ==========================================

total = len(evaluation_results)

tool_passed = sum(
    1 for result in evaluation_results
    if result["tool_correctness"]["passed"]
)

relevancy_evaluated = [
    result["answer_relevancy"]
    for result in evaluation_results
    if result["answer_relevancy"]["passed"] is not None
]

faithfulness_evaluated = [
    result["faithfulness"]
    for result in evaluation_results
    if result["faithfulness"]["passed"] is not None
]

print("\n" + "=" * 60)
print("FINAL DEEPEVAL EVALUATION REPORT")
print("=" * 60)

print("Total Test Cases:", total)

print(
    "Tool Correctness:",
    f"{tool_passed}/{total}"
)

print(
    "Answer Relevancy:",
    f"{sum(1 for x in relevancy_evaluated if x['passed'])}"
    f"/{len(relevancy_evaluated)} evaluated"
)

print(
    "Faithfulness:",
    f"{sum(1 for x in faithfulness_evaluated if x['passed'])}"
    f"/{len(faithfulness_evaluated)} evaluated"
)

print("=" * 60)