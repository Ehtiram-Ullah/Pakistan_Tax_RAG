

from retrieval import search

test_case = [
    {
        "question": "What is the tax on a motor vehicle above 3000cc?",
        "expected_pages": [578, 579]
    },
    {
        "question": "What is the tax on a motor vehicle between 1601cc and 1800cc?",
        "expected_pages": [578, 579]
    },
]

for test in test_case:
    results = search(test["question"],limit=5)

    retrieved_pages = [
        result.payload["page"]
        for result in results
    ]

    print("\nQUESTION:")
    print(test["question"])

    print("Expected pages:", test["expected_pages"])
    print("Retrieved pages:", retrieved_pages)