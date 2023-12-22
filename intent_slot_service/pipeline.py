import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class ChatBot:
    def __init__(self, model_path, max_length=256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.max_length = max_length
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)

    def generate_answer(self, question):
        source_encoding = self.tokenizer.encode_plus(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=source_encoding["input_ids"],
                attention_mask=source_encoding["attention_mask"],
                max_length=10, 
                return_dict_in_generate=True,
                output_scores=True
            )
            result = self.tokenizer.decode(generated["sequences"][0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return result

    def __call__(self, question):
        return self.generate_answer(question)

if __name__ == '__main__':
    model_path = ""
    chatbot = ChatBot(model_path)
    question = "Trường có chương trình hỗ trợ học bổng cho học sinh xuất sắc không?"
    result = chatbot(question)
    print(result)