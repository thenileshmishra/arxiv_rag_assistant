from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# -------------------------------------------------------------------
# Load environment variables from .env in project root (safe default)
# -------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
dotenv_path = project_root / ".env"

if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    load_dotenv()  # fallback to current dir

# Load and set environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not GOOGLE_API_KEY:
    print("[WARN] GOOGLE_API_KEY not found in environment. Gemini mode may not work.", file=sys.stderr)
if not OPENAI_API_KEY:
    print("[INFO] OPENAI_API_KEY not found in environment. Skipping OpenAI unless set.", file=sys.stderr)


class LLM:
    """
    A unified client for interacting with different Large Language Models (LLMs).

    Supported modes:
      - "hf"     : Hugging Face Transformers (local inference)
      - "gemini" : Google Gemini API
      - "openai" : OpenAI API
    """

    def __init__(self, mode: str = "gemini", model_name: str = "gemini-1.5-flash",
                 temperature: float = 0.0, max_tokens: int = 512):
        """
        Initialize the LLM client.

        Args:
            mode (str): The LLM provider ('hf', 'gemini', or 'openai').
            model_name (str): Model identifier for the provider.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens to generate.
        """
        self.mode = mode
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.mode == "hf":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
                self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device_map="auto")
            except ImportError:
                raise ImportError("Hugging Face requires 'transformers'. Install: pip install transformers")

        elif self.mode == "gemini":
            try:
                import google.generativeai as genai
                google_key = os.environ.get("GOOGLE_API_KEY")

                if not google_key:
                    raise RuntimeError("GOOGLE_API_KEY environment variable not set. Cannot use Gemini.")

                genai.configure(api_key=google_key)
                self.gen_model = genai.GenerativeModel(model_name=self.model_name)

            except ImportError:
                raise ImportError("Gemini requires 'google-generativeai'. Install: pip install google-generativeai")

        elif self.mode == "openai":
            try:
                import openai
                if not OPENAI_API_KEY:
                    raise RuntimeError("OPENAI_API_KEY not set. Cannot use OpenAI.")
                self.openai = openai
                self.openai.api_key = OPENAI_API_KEY
            except ImportError:
                raise ImportError("OpenAI requires 'openai'. Install: pip install openai")

        else:
            raise ValueError("Unsupported mode. Choose from: 'hf', 'gemini', 'openai'.")

    def generate(self, prompt: str) -> str:
        """
        Generate text from the model.

        Args:
            prompt (str): Input query.

        Returns:
            str: Model output.
        """
        try:
            if self.mode == "hf":
                outputs = self.pipe(prompt, max_new_tokens=self.max_tokens, do_sample=False)
                return outputs[0]["generated_text"].strip()

            elif self.mode == "gemini":
                resp = self.gen_model.generate_content(prompt)
                return resp.text.strip() if resp and resp.text else "[Empty Gemini response]"

            elif self.mode == "openai":
                resp = self.openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return resp["choices"][0]["message"]["content"].strip()

        except Exception as e:
            return f"[{self.mode.upper()} ERROR] {e}"


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        gemini_client = LLM(mode="gemini", model_name="gemini-1.5-flash")
        print("Gemini:", gemini_client.generate("What is the capital of France?"))

        # Uncomment if you have OPENAI_API_KEY set
        # openai_client = LLM(mode="openai", model_name="gpt-4o-mini")
        # print("OpenAI:", openai_client.generate("What is the capital of France?"))

    except Exception as e:
        print(f"[INIT ERROR] {e}", file=sys.stderr)
