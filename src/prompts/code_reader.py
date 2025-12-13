code_reader_system_prompt = "You are a code summarizer. You receive a file path. Use the provided file content to give a concise summary of what the code does."


def code_reader_user_prompt(filepath: str, file_content: str) -> str:
    return f"Path: {filepath}\n\n{file_content}"
