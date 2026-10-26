from agent import agent

test_cases = [
    {
        "question": "Calculate 125 multiplied by 48.",
        "expected_tool": "multiply_numbers",
        "expected_output": "6000"
    },
    {
        "question": "Count the words and characters in Agentic AI is the future.",
        "expected_tool": "text_analyzer",
        "expected_output": "5"
    },
    {
        "question": "Explain LangChain using knowledge search.",
        "expected_tool": "knowledge_search",
        "expected_output": "LangChain"
    },
    {
        "question": "Use RAG to explain Agentic AI.",
        "expected_tool": "rag_search",
        "expected_output": "Agentic AI"
    }
]

passed = 0

for case in test_cases:

    print("\n" + "=" * 60)
    print("QUESTION:", case["question"])

    try:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": case["question"]
                }
            ]
        })

        actual_tools = []

        for message in response["messages"]:
            for call in getattr(message, "tool_calls", []):
                actual_tools.append(call["name"])

        answer = str(response["messages"][-1].content)

        tool_correct = case["expected_tool"] in actual_tools

        # Basic answer validation
        answer_correct = (
            case["expected_output"].lower()
            in answer.lower()
        )

        success = tool_correct and answer_correct

        if success:
            passed += 1

        print("Expected Tool:", case["expected_tool"])
        print("Actual Tools:", actual_tools)
        print("Agent Answer:", answer)
        print("Tool Correct:", tool_correct)
        print("Answer Correct:", answer_correct)
        print("Status:", "PASS" if success else "FAIL")

    except Exception as e:
        print("ERROR:", str(e))
        print("Status: FAIL")

total = len(test_cases)

accuracy = (passed / total) * 100 if total else 0

print("\n" + "=" * 60)
print("FINAL TOOL EXECUTION REPORT")
print("=" * 60)

print("Total Tests:", total)
print("Passed:", passed)
print("Failed:", total - passed)
print(f"Tool Execution Accuracy: {accuracy:.2f}%")

print("=" * 60)