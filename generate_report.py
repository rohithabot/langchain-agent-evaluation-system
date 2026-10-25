import json
from pathlib import Path
from datetime import datetime

# Step 1: Define file paths
BASE_DIR = Path(__file__).resolve().parent

input_path = BASE_DIR / "evaluation_results.json"
output_path = BASE_DIR / "final_evaluation_report.md"

print("Starting Agentic AI Evaluation Report Generation...")

# Step 2: Load evaluation results
if not input_path.exists():
    raise FileNotFoundError(
        f"Evaluation results file not found: {input_path}"
    )

with open(input_path, "r", encoding="utf-8") as file:
    results = json.load(file)

print("Evaluation results loaded successfully!")

# Step 3: Calculate results
total = len(results)

tool_passed = 0
relevancy_passed = 0
faithfulness_passed = 0

relevancy_total = 0
faithfulness_total = 0

for result in results:

    tool = result.get("tool_correctness") or {}
    relevancy = result.get("answer_relevancy") or {}
    faithfulness = result.get("faithfulness") or {}

    if tool.get("passed") is True:
        tool_passed += 1

    if relevancy.get("passed") is not None:
        relevancy_total += 1

        if relevancy["passed"]:
            relevancy_passed += 1

    if faithfulness.get("passed") is not None:
        faithfulness_total += 1

        if faithfulness["passed"]:
            faithfulness_passed += 1


def calculate_accuracy(passed, total):
    return (passed / total * 100) if total else 0


# Step 4: Generate report
report = f"""# Agentic AI Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview

This project evaluates a LangChain Agentic AI system
using Ollama and DeepEval.

## Evaluation Results

| Metric | Passed | Evaluated | Accuracy |
|---|---:|---:|---:|
| Tool Correctness | {tool_passed} | {total} | {calculate_accuracy(tool_passed, total):.2f}% |
| Answer Relevancy | {relevancy_passed} | {relevancy_total} | {calculate_accuracy(relevancy_passed, relevancy_total):.2f}% |
| Faithfulness | {faithfulness_passed} | {faithfulness_total} | {calculate_accuracy(faithfulness_passed, faithfulness_total):.2f}% |

## Conclusion

The system was evaluated using {total} test cases.

Tool correctness passed {tool_passed} tests.

Answer relevancy passed {relevancy_passed} evaluated tests.

Faithfulness passed {faithfulness_passed} evaluated tests.

These results reflect the tested scenarios and do not guarantee
performance on unseen questions.
"""

# Step 5: Save report
with open(output_path, "w", encoding="utf-8") as file:
    file.write(report)

# Step 6: Display results
print("\n" + "=" * 60)
print("FINAL AGENTIC AI EVALUATION REPORT")
print("=" * 60)

print("Total Test Cases:", total)
print(f"Tool Correctness: {tool_passed}/{total}")
print(f"Answer Relevancy: {relevancy_passed}/{relevancy_total}")
print(f"Faithfulness: {faithfulness_passed}/{faithfulness_total}")

print("\nReport generated successfully!")
print("Saved to:", output_path)

print("=" * 60)