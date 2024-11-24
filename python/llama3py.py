import torch
import torchtune.models as models
import torchtune.models.phi3 as tokenizer

model = models.phi3()  # Loads the Phi-3 Mini model
# tokenizer = phi3_mini_tokenizer()

text = "This is an example sentence."
input_ids = tokenizer.encode(text)


input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
output = model.generate(input_ids)
generated_text = tokenizer.decode(output[0])

print(generated_text)
