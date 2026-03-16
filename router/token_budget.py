"""
Token Budget Manager.
Tracks token usage and accumulated cost to enforce budget limits.
"""

class TokenBudgetManager:
    def __init__(self, initial_budget_usd: float = 5.0):
        self.initial_budget_usd = initial_budget_usd
        self.spent_usd = 0.0
        
        # Approximate costs per 1K tokens (Input, Output) in USD
        self.costs = {
            "claude-opus": (0.015, 0.075),
            "claude-sonnet": (0.003, 0.015),
            "claude-haiku": (0.00025, 0.00125),
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006),
            "mistral-small": (0.0002, 0.0006),
            "ollama": (0.0, 0.0) # Local models are free
        }
        
    def track_usage(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Updates the spent budget and returns the cost of this particular call.
        """
        model_cost = self.costs.get(model, (0.0, 0.0))
        cost = (input_tokens / 1000.0) * model_cost[0] + (output_tokens / 1000.0) * model_cost[1]
        self.spent_usd += cost
        print(f"[Budget] Spent ${cost:.4f} on {model}. Total spent: ${self.spent_usd:.4f}")
        return cost
        
    def get_remaining_budget(self) -> float:
        return max(0.0, self.initial_budget_usd - self.spent_usd)
        
    def should_downgrade(self, current_model: str) -> bool:
        """
        Returns True if remaining budget is less than 20% of the initial budget,
        flagging that we should use a cheaper model.
        """
        remaining = self.get_remaining_budget()
        ratio = remaining / self.initial_budget_usd
        
        # Downgrade if < 20% remaining budget, but don't downgrade free local models
        if ratio < 0.20 and current_model != "ollama":
            return True
        return False

if __name__ == "__main__":
    budget = TokenBudgetManager(initial_budget_usd=1.0)
    budget.track_usage("claude-opus", 5000, 2000)
    print(f"Remaining: ${budget.get_remaining_budget():.4f}")
    print(f"Should downgrade opus? {budget.should_downgrade('claude-opus')}")
