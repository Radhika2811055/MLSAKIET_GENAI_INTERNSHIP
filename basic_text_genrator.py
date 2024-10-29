from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# above libraries are necessary for the implementation

# Load pre-trained model 
model_name = "gpt2"
# use pre-trained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()  #it's work is to evaluate

# this function generates relevant output to your input text.
def generate_text(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            #it prevents that the lines are not repeated
            no_repeat_ngram_size=2,  
            # after recahing the max limit the code should stop
            early_stopping=True
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# here the user will enter the prompt
prompt = input("Enter the text: ")
# it is the generated output dsiplayed containing the output text similar to the text input.
generated_output = generate_text(prompt, max_length=100)
print("\n")
print("generated text: ",generated_output)