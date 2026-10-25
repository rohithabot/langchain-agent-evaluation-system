from agent import agent

# Step 1: Define test questions and expected tools
test_cases = [
    {
        "question": "Calculate 125 multiplied by 48.",
        "expected_tool": "multiply_numbers"
    },
    {
        "question": "Count the words and characters in Agentic AI is the future.",
        "expected_tool": "text_analyzer"
    },
    {
        "question": "Explain LangChain using knowledge search.",
        "expected_tool": "knowledge_search"
    },
    {
        "question": "Use RAG to explain Agentic AI.",
        "expected_tool": "rag_search"
    }
]

# Step 2: Execute the real agent
passed = 0
results = []

for case in test_cases:

    question = case["question"]
    expected_tool = case["expected_tool"]

    print("\n" + "=" * 60)
    print("QUESTION:", question)

    try:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

        # Step 3: Capture actual tool calls
        actual_tools = []

        for message in response["messages"]:
            for call in getattr(message, "tool_calls", []):
                actual_tools.append(call["name"])

        # Step 4: Compare expected and actual tools
        success = expected_tool in actual_tools

        if success:
            passed += 1

        print("Expected Tool:", expected_tool)
        print("Actual Tools:", actual_tools)
        print("Status:", "PASS" if success else "FAIL")
        print("Agent Answer:", response["messages"][-1].content)

        results.append({
            "question": question,
            "expected_tool": expected_tool,
            "actual_tools": actual_tools,
            "passed": success
        })

    except Exception as e:
        print("ERROR:", str(e))
        results.append({
            "question": question,
            "expected_tool": expected_tool,
            "actual_tools": [],
            "passed": False,
            "error": str(e)
        })

# Step 5: Calculate actual accuracy
total = len(test_cases)
failed = total - passed
accuracy = (passed / total) * 100 if total else 0

print("\n" + "=" * 60)
print("FINAL REAL AGENT EVALUATION REPORT")
print("=" * 60)

print("Total Test Cases:", total)
print("Passed:", passed)
print("Failed:", failed)
print(f"Real Tool Selection Accuracy: {accuracy:.2f}%")

print("=" * 60)