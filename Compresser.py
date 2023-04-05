from transformers import AutoTokenizer, AutoModelWithLMHead

class Compresser:
    def __init__(self, text):
        self.text = text
        self.mask_token = '[MASK]' # this is the token that will be replaced by the model
        self.summary = None

    def summarize(self):
        # Use GPT to generate a summary
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelWithLMHead.from_pretrained("gpt2")

        input_ids = tokenizer.encode(self.text, return_tensors="pt")
        summary_ids = model.generate(input_ids, max_length=100, min_length=30, num_beams=2, no_repeat_ngram_size=2)

        # Convert summary IDs to text
        self.summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

text = "Produce a short summary on this"
compresser = Compresser(text)
compresser.summarize()
print(compresser.summary)
