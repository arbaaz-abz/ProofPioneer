import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from itertools import cycle
import json
import time

class GeminiAPI:
    def __init__(self, model_name, temperature=0.25, top_p=0.9, top_k=40, response_mime_type="text/plain"):
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "response_mime_type": response_mime_type,
        }
        if "flash" in model_name:
            self.max_requests_per_key = 1500
            self.sleep_time = 3.75
        else:
            self.max_requests_per_key = 50
            self.sleep_time = 30
            
        self.model_name = model_name
        self.request_count = 0

        with open("ProofPioneer/secrets/gemini_keys.json", "r") as f:
            self.api_keys = cycle(json.load(f)["keys"])

        self.get_model = self.get_gemini_model()
        self.model = self.get_model()

    def get_gemini_model(self):
        def initialize_model():
            current_key = next(self.api_keys)
            print("Using key: ****" + current_key[-5:])
            genai.configure(api_key=current_key)
            return genai.GenerativeModel(model_name=self.model_name, generation_config=self.generation_config)
        return initialize_model

    def get_llm_response(self, input_text, chat=None, reset=True):
        if self.request_count >= self.max_requests_per_key:
            self.model = self.get_model()
            self.request_count = 0

        if reset:
            chat = self.model.start_chat(history=[])
        try:
            resp = chat.send_message(content=input_text, 
                             safety_settings={
                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                             }, 
                             generation_config=self.generation_config)
            time.sleep(self.sleep_time)
            self.request_count += 1
            return resp
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(self.sleep_time)
            return None

# Usage example:
# gemini_api = GeminiAPI()
# response = gemini_api.get_llm_response("Your query here")