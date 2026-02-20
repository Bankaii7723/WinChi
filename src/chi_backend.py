from ctransformers import AutoModelForCausalLM
import os

class ChiBackend:
    """
    Backend for phi-3-mini GGUF models using ctransformers AutoModelForCausalLM
    """

    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # create the model using ctransformers’ unified loader
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    def gentxt(self, prompt: str, tokens: int = 256, temp: float = 0.7, experimental_streaming: bool = True):
        try:
            output = self.model(prompt, max_tokens=tokens, temperature=temp)
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")

        if experimental_streaming:
            for ch in output:
                yield ch
        else:
            return output
