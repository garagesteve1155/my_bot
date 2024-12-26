import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from PIL import Image
import requests

class Session:
    def __init__(self, model_name):
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load the model with 8-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,        # Enable 8-bit quantization
            device_map="auto",        # Automatically map to GPU
            torch_dtype=torch.float16 # Use mixed precision
        )
        self.model.eval()  # Set model to evaluation mode
        self.model.to("cuda")  # Move model to GPU
        
        # Initialize session history
        self.history = []
    
    def add_to_history(self, user_input, model_response):
        """Add a user input and model response to the session history."""
        self.history.append({"user": user_input, "model": model_response})
    
    def get_history_text(self):
        """Convert session history to a formatted text string."""
        history_text = ""
        for exchange in self.history:
            history_text += f"User: {exchange['user']}\n"
            history_text += f"Model: {exchange['model']}\n"
        return history_text
    
    def generate_response(self, user_input, image_path=None):
        """
        Generate a response from the model given user input and optional image.
        Maintains session history for context.
        """
        # Prepare session history
        prompt = self.get_history_text() + f"User: {user_input}\nModel:"
        
        # Tokenize text
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # If an image is provided, process it
        if image_path:
            image = Image.open(image_path).convert("RGB")
            image_inputs = self.processor(images=image, return_tensors="pt").to("cuda")
        else:
            image_inputs = None
        
        # Generate response
        with torch.no_grad():
            if image_inputs:
                outputs = self.model.generate(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    images=image_inputs["pixel_values"],
                    max_length=500,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            else:
                outputs = self.model.generate(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    max_length=500,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the model's response
        # Assuming the model continues from "Model:"
        response = generated_text.split("Model:")[-1].strip()
        
        # Add to history
        self.add_to_history(user_input, response)
        
        return response

def main():
    # Define model name
    model_name = "meta-llama/Llama-3.2-Vision-Instruct-11B"  # Replace with the actual model ID
    
    # Initialize session
    session = Session(model_name)
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if the user wants to exit
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the session.")
            break
        
        # Optional: Ask if the user wants to provide an image
        include_image = input("Do you want to include an image? (y/n): ").lower()
        if include_image == 'y':
            image_path = input("Enter the image file path: ")
        else:
            image_path = None
        
        # Generate model response
        response = session.generate_response(user_input, image_path=image_path)
        
        # Print the response
        print(f"Model: {response}\n")

if __name__ == "__main__":
    main()