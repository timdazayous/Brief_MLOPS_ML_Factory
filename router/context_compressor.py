"""
Context Compressor.
Compresses prompts that exceed a maximum token limit before sending them to models.
Preserves critical information like metrics, errors, and numerical values.
"""

def compress_if_needed(prompt: str, max_tokens: int = 2000) -> str:
    """
    Simulates token counting and compression.
    If the prompt is too long, we extract key parts or use a local/lightweight 
    model to summarize it before the main LLM call.
    """
    # Rough approximation: 1 token = 4 characters
    approx_tokens = len(prompt) // 4
    
    if approx_tokens <= max_tokens:
        return prompt
        
    print(f"[Compressor] Prompt exceeds {max_tokens} tokens ({approx_tokens} approx). Compressing...")
    
    # In a real app, you would call a cheap model like Haiku or Mistral-small here
    # to summarize the text while preserving "metrics, errors, numerical values".
    # For now, we simulate compression by keeping the first and last parts.
    
    keep_chars = (max_tokens * 4) // 2
    compressed_prompt = (
        prompt[:keep_chars] + 
        "\n\n... [CONTENT COMPRESSED FOR TOKEN LIMITS] ...\n\n" + 
        prompt[-keep_chars:]
    )
    
    return compressed_prompt

if __name__ == "__main__":
    long_prompt = "A" * 10000
    compressed = compress_if_needed(long_prompt, max_tokens=1000)
    print(f"Original length: {len(long_prompt)}")
    print(f"Compressed length: {len(compressed)}")
