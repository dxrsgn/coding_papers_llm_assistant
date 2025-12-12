from langchain_core.prompts import ChatPromptTemplate


code_reader_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a code summarizer. You receive a file path. Use the provided file content to give a concise summary of what the code does.",
        ),
        ("human", "Path: {filepath}\n\n{file_content}"),
    ]
)
