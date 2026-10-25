import json
from pathlib import Path

from langchain_core.messages import ToolMessage

from agent import agent


# ==========================================
# STEP 1: Define test questions
# ==========================================

test_cases = [
    {
        "question": "Calculate 125 multiplied by 48.",
        "expected_tool": "multiply_numbers"
    },
    {
        "question": (
            "Count the words and characters in "
            "'Agentic AI is the future' using the text analyzer."
        ),
        "expected_tool": "text_analyzer"
    },
    {
        "question": "Use knowledge_search to explain LangChain.",
        "expected_tool": "knowledge_search"
    },
    {
        "question": "Use RAG to explain Agentic AI.",
        "expected_tool": "rag_search"
    }
]


# ==========================================
# STEP 2: Execute real agent
# ==========================================

results = []

retrieval_tools = {
    "knowledge_search",
    "rag_search"
}

for index, case in enumerate(test_cases, start=1):

    question = case["question"]

    print("\n" + "=" * 60)
    print(f"TEST CASE {index}")
    print("=" * 60)

    print("Question:", question)

    try:

        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

        # Capture final response
        answer = response["messages"][-1].content

        # Capture actual tool calls
        tools_used = []

        # Map tool call IDs to tool names
        tool_call_names = {}

        for message in response["messages"]:

            for call in getattr(message, "tool_calls", []):

                tools_used.append({
                    "name": call["name"],
                    "args": call["args"]
                })

                tool_call_names[call["id"]] = call["name"]

        actual_tool_names = [
            tool["name"] for tool in tools_used
        ]

        # ==================================
        # Capture retrieved context
        # ==================================

        retrieval_context = []

        for message in response["messages"]:

            if isinstance(message, ToolMessage):

                tool_name = (
                    getattr(message, "name", None)
                    or tool_call_names.get(
                        message.tool_call_id
                    )
                )

                if (
                    tool_name in retrieval_tools
                    and message.status != "error"
                ):

                    context = message.content

                    if isinstance(context, str):

                        retrieval_context.append(context)

        # ==================================
        # Compare expected and actual tools
        # ==================================

        passed = (
            case["expected_tool"] in actual_tool_names
        )

        print("\nAGENT ANSWER:")
        print(answer)

        print("\nEXPECTED TOOL:")
        print(case["expected_tool"])

        print("\nACTUAL TOOLS:")
        print(actual_tool_names)

        print("\nRETRIEVED CONTEXT:")

        if retrieval_context:

            for context in retrieval_context:
                print(context)

        else:
            print("No retrieval context")

        print("\nSTATUS:", "PASS" if passed else "FAIL")

        results.append({
            "question": question,
            "expected_tool": case["expected_tool"],
            "actual_tools": actual_tool_names,
            "tool_calls": tools_used,
            "answer": answer,
            "retrieval_context": retrieval_context,
            "passed": passed
        })

    except Exception as error:

        print("ERROR:", str(error))

        results.append({
            "question": question,
            "expected_tool": case["expected_tool"],
            "actual_tools": [],
            "tool_calls": [],
            "answer": "",
            "retrieval_context": [],
            "passed": False,
            "error": str(error)
        })


# ==========================================
# STEP 3: Save results as JSON
# ==========================================

output_path = (
    Path(__file__).resolve().parent
    / "agent_results.json"
)

with open(output_path, "w", encoding="utf-8") as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ==========================================
# STEP 4: Print final summary
# ==========================================

total = len(results)

passed = sum(
    1 for result in results
    if result["passed"]
)

failed = total - passed

accuracy = (
    (passed / total) * 100
    if total else 0
)

print("\n" + "=" * 60)
print("FINAL AGENT TEST REPORT")
print("=" * 60)

print("Total Tests:", total)
print("Passed:", passed)
print("Failed:", failed)

print(
    f"Tool Selection Accuracy: {accuracy:.2f}%"
)

print("\nResults saved to:")
print(output_path)

print("=" * 60)