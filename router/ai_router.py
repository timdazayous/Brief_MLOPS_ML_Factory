"""
Intelligent AI Router.
Combines skill classification, token budgeting, and context compression to route requests.
Has a graceful fallback chain if the primary model fails.
"""

import mlflow
from pydantic import BaseModel
from typing import Optional

# Import other router components
from router.skill_classifier import classify_task, TaskType
from router.token_budget import TokenBudgetManager
from router.context_compressor import compress_if_needed

class ModelResponse(BaseModel):
    model_used: str
    response: str
    cost: float

class AIRouter:
    def __init__(self, budget_usd: float = 5.0):
        self.budget = TokenBudgetManager(initial_budget_usd=budget_usd)
        
        # Skill to Model mapping
        self.routing_table = {
            TaskType.REASONING: "claude-opus",
            TaskType.CODE: "claude-sonnet",
            TaskType.VALIDATION: "claude-haiku",
            TaskType.SENSITIVE: "ollama"
        }
        
        # Fallback chain for resilience
        self.fallback_chain = {
            "claude-opus": ["claude-sonnet", "gpt-4o", "mistral-small", "ollama"],
            "claude-sonnet": ["gpt-4o-mini", "claude-haiku", "ollama"],
            "claude-haiku": ["mistral-small", "ollama"],
            "gpt-4o": ["gpt-4o-mini", "ollama"],
            "gpt-4o-mini": ["claude-haiku", "ollama"],
            "mistral-small": ["ollama"],
            "ollama": [] # No fallback for local
        }

    def _mock_call(self, model: str, prompt: str) -> str:
        """
        Mock function to simulate an LLM call. 
        In production, replace with litellm, requests, or specific SDKs.
        """
        # Simulate a local failure occasionally or if model is missing
        # if model == "claude-opus":
        #    raise Exception("API Timeout")
        return f"[{model}] Response to: {prompt[:50]}..."

    def _execute_with_fallback(self, initial_model: str, prompt: str) -> tuple[str, str]:
        """
        Tries to execute the prompt with the initial model. If it fails,
        it walks down the fallback chain.
        Returns (successful_model, response)
        """
        models_to_try = [initial_model] + self.fallback_chain.get(initial_model, [])
        
        for model in models_to_try:
            try:
                print(f"[Router] Attempting to call {model}...")
                response = self._mock_call(model, prompt)
                return model, response
            except Exception as e:
                print(f"[Router] {model} failed: {e}. Trying next fallback...")
                continue
                
        raise Exception("All models in the fallback chain failed.")

    def route(self, prompt: str, task_hint: Optional[TaskType] = None) -> ModelResponse:
        """
        Main routing logic: Classify -> Budget -> Compress -> Call -> Track -> Log
        """
        # 1. Classify Task
        task_type = task_hint if task_hint else classify_task(prompt)
        print(f"[Router] Task classified as: {task_type.value}")
        
        # 2. Select Initial Model
        target_model = self.routing_table.get(task_type, "claude-haiku")
        
        # 3. Check Budget & Downgrade if needed
        if self.budget.should_downgrade(target_model):
            print(f"[Router] Budget low (<20%). Downgrading from {target_model}...")
            # Use the first fallback as the new target
            if self.fallback_chain.get(target_model):
                target_model = self.fallback_chain[target_model][0]
                
        # 4. Context Compression
        compressed_prompt = compress_if_needed(prompt, max_tokens=2000)
        
        # 5. Execute with Fallback
        used_model, response_text = self._execute_with_fallback(target_model, compressed_prompt)
        
        # 6. Track Usage
        # Simulate token counts based on length
        in_tokens = len(prompt) // 4
        out_tokens = len(response_text) // 4
        cost = self.budget.track_usage(used_model, in_tokens, out_tokens)
        
        # 7. Log to MLflow
        try:
            if mlflow.active_run():
                mlflow.log_metric(f"router_cost_{used_model}", cost)
                mlflow.log_param("last_router_task", task_type.value)
        except Exception:
            pass # Ignore if no active run
            
        return ModelResponse(model_used=used_model, response=response_text, cost=cost)

if __name__ == "__main__":
    router = AIRouter(budget_usd=1.0)
    
    # Test reasoning
    resp1 = router.route("What is the optimal architecture for this ML pipeline?")
    print(resp1)
    
    # Test sensitive data
    resp2 = router.route("My client's private key is 12345.")
    print(resp2)
