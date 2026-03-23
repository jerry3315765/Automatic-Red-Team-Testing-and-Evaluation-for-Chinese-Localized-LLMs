import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import requests
import json

class LocalLLM:
    def __init__(self, config):
        self.config = config
        self.model_path = config.get('path')
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.quantization = config.get('quantization', 'none')
        self.lm_studio_url = config.get('lm_studio_url', 'http://localhost:1234')

        print(f"Loading model: {self.config.get('name')} from {self.model_path}")
        self._load_model()

    def _load_model(self):
        try:
            # Check for LM Studio triggers (gguf, Q8, MXFP4)
            if self.quantization in ['gguf', 'Q8', 'MXFP4']:
                # For GGUF/LM Studio models, we'll use LM Studio API
                print(f"LM Studio model detected (Quantization: {self.quantization}) - will use LM Studio API")
                self.use_lm_studio = True

                # Try to load model via cli if available
                import subprocess
                if self.model_path:
                    try:
                        print(f"Attempting to load model {self.model_path} via 'lms load'...")
                        # Run lms load command
                        subprocess.run(f"lms load {self.model_path}", shell=True, check=True)
                        print("Model load command sent successfully")
                    except Exception as e:
                        print(f"Warning: Failed to load model via 'lms load': {e}")
                        print("Please ensure the model is loaded in LM Studio manually if generation fails.")

                # Test connection
                try:
                    response = requests.get(f"{self.lm_studio_url}/v1/models", timeout=5)
                    if response.status_code == 200:
                        print("LM Studio connection successful")
                    else:
                        print(f"LM Studio connection failed: {response.status_code}")
                        raise Exception("LM Studio not available")
                except Exception as e:
                    print(f"Cannot connect to LM Studio: {e}")
                    raise e
            else:
                # Original transformers loading for non-GGUF models
                self.use_lm_studio = False
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                bnb_config = None
                if self.quantization == '8bit':
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                elif self.quantization == '4bit':
                    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

                load_args = {
                    "trust_remote_code": True,
                    "device_map": "auto" if self.device == 'cuda' else None,
                }

                if bnb_config:
                    load_args["quantization_config"] = bnb_config
                    print(f"Quantization enabled: {self.quantization}")
                elif self.device == 'cuda':
                    load_args["torch_dtype"] = torch.bfloat16

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **load_args
                )
                self.model.eval()
                print("Model loaded successfully.")

        except Exception as e:
            print(f"Error loading model {self.config.get('name')}: {e}")
            raise e

    def generate(self, prompts):
        results = []
        for item in tqdm(prompts, desc=f"Generating with {self.config.get('name')}"):
            # Handle prompt as string or list of messages
            prompt_input = item['prompt']
            
            # Prepare messages list for chat models
            if isinstance(prompt_input, str):
                messages = [{"role": "user", "content": prompt_input}]
            else:
                messages = prompt_input
            
            if self.use_lm_studio:
                # Use LM Studio API for GGUF models
                try:
                    payload = {
                        "model": self.model_path if self.model_path else "local-model",
                        "messages": messages,
                        "max_tokens": 2048,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stream": False
                    }

                    response = requests.post(
                        f"{self.lm_studio_url}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=120
                    )

                    if response.status_code == 200:
                        data = response.json()
                        response_text = data['choices'][0]['message']['content']
                    else:
                        response_text = f"API Error: {response.status_code} - {response.text}"

                except Exception as e:
                    response_text = f"LM Studio Error: {str(e)}"

            else:
                # Original transformers generation
                try:
                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback if no chat template or raw string
                    if isinstance(prompt_input, str):
                        input_text = prompt_input
                    else:
                         # Simple join if list but no template
                         input_text = "\n".join([m['content'] for m in messages])

                inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=2048,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Store original prompt/messages in result for reference
            results.append({
                "model": self.config.get('name'),
                "template_id": item['template_id'],
                "intent": item['intent'],
                "prompt": prompt_input if isinstance(prompt_input, str) else json.dumps(prompt_input, ensure_ascii=False),
                "response": response_text
            })

        return results
