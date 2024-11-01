import tiktoken


class CostAnalyzer:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        input_price_per_1m_tokens=0.150,
        output_price_per_1m_tokens=0.600,
    ):
        self.model_name = model_name
        self.input_price_per_1m_tokens = input_price_per_1m_tokens
        self.output_price_per_1m_tokens = output_price_per_1m_tokens
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_input_cost = 0
        self.total_output_cost = 0
        self.total_verification_tokens = 0
        self.total_verification_cost = 0

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def calculate_cost(self, input_tokens, output_tokens):
        input_cost = (input_tokens / 1_000_000) * self.input_price_per_1m_tokens
        output_cost = (output_tokens / 1_000_000) * self.output_price_per_1m_tokens
        return input_cost, output_cost

    def add_usage(self, input_tokens, output_tokens):
        input_cost, output_cost = self.calculate_cost(input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_input_cost += input_cost
        self.total_output_cost += output_cost
        return input_cost, output_cost

    def add_verification_usage(self, input_tokens, output_tokens):
        input_cost, output_cost = self.calculate_cost(input_tokens, output_tokens)
        self.total_verification_tokens += input_tokens + output_tokens
        self.total_verification_cost += input_cost + output_cost
        return input_cost + output_cost

    def get_summary(self):
        summary = {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_input_cost": self.total_input_cost,
            "total_output_cost": self.total_output_cost,
            "total_cost": self.total_input_cost + self.total_output_cost,
            "total_verification_tokens": self.total_verification_tokens,
            "total_verification_cost": self.total_verification_cost,
            "grand_total_cost": self.total_input_cost
            + self.total_output_cost
            + self.total_verification_cost,
        }
        return summary
