"""
Skill Classifier for AI tasks.
Categorizes prompts into REASONING, CODE, VALIDATION, or SENSITIVE.
"""

from enum import Enum

class TaskType(Enum):
    REASONING = "REASONING"
    CODE = "CODE"
    VALIDATION = "VALIDATION"
    SENSITIVE = "SENSITIVE"

SENSITIVE_KEYWORDS = ["password", "secret", "private key", "ssn", "credit card", "confidential"]
CODE_KEYWORDS = ["code", "script", "python", "function", "bug", "refactor", "generate"]
VALIDATION_KEYWORDS = ["validate", "parse", "log", "summary", "summarize", "check", "explain"]

def classify_task(prompt: str) -> TaskType:
    """
    Classifies the given prompt into a TaskType based on heuristics.
    In a real-world scenario, you could also use an LLM call here.
    """
    lower_prompt = prompt.lower()
    
    # 1. Check for sensitive data
    if any(kw in lower_prompt for kw in SENSITIVE_KEYWORDS):
        return TaskType.SENSITIVE
        
    # 2. Check for code-related tasks
    if any(kw in lower_prompt for kw in CODE_KEYWORDS):
        return TaskType.CODE
        
    # 3. Check for validation / summarization
    if any(kw in lower_prompt for kw in VALIDATION_KEYWORDS):
        return TaskType.VALIDATION
        
    # 4. Default to reasoning
    return TaskType.REASONING

if __name__ == "__main__":
    # Examples
    print(classify_task("Please debug this python script.")) # CODE
    print(classify_task("Summarize the latest MLflow logs.")) # VALIDATION
    print(classify_task("What is the optimal architecture for this dataset?")) # REASONING
    print(classify_task("My password is test.")) # SENSITIVE
