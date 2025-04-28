import os
from litellm import completion
gemini_api_key = os.getenv("GEMINI_API_KEY")

def generate_response(query, context, provider):
    prompt = f"Query: {query}\nContext: {context}\nAnswer:"
    if provider == "huggingface":
        api_key = os.getenv("HUGGINGFACE_TOKEN")
        model = "huggingface/HuggingFaceH4/zephyr-7b-beta"  # HuggingFace Zephyr
        response = completion(
            model=model,
            messages=[{"content": prompt, "role": "user"}],
            api_key=api_key,
        )
    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = "llama3-70b-8192"  # Groq Llama3 70B
        response = completion(
            model=model,
            messages=[{"content": prompt, "role": "user"}],
            api_key=api_key,
            api_base="https://api.groq.com/openai/v1",  # Groq needs this
        )
    else:
        response = completion(
            model="gemini/gemini-1.5-flash",
            messages=[{"content": prompt, "role": "user"}],
            api_key=gemini_api_key,
        )
    return response['choices'][0]['message']['content']