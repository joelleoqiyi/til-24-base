from typing import Dict
# ADDED:
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import re

number_map = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9"
}

class NLPManager:
    def __init__(self):
        # initialize the model here
        # START CODE
        self.tokenizer = AutoTokenizer.from_pretrained("valhalla/electra-base-discriminator-finetuned_squadv1")
        self.model = AutoModelForQuestionAnswering.from_pretrained("valhalla/electra-base-discriminator-finetuned_squadv1")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device) # runs model on gpu
        # END CODE
        pass

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        # START CODE
        questions = {
            'heading': "What is the heading value?", 
            'target': "What is the full target description?",
            'tool': "What is the tool to deploy?"
        }
        
        answers = {}
        
        for key, question in questions.items():
            inputs = self.tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
            inputs = inputs.to(self.device) # run on GPU
            input_ids = inputs["input_ids"].tolist()[0]

            text_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            #print(text_tokens)
            outputs = self.model(**inputs)
            answer_start_scores=outputs.start_logits
            answer_end_scores=outputs.end_logits
            
            answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            
            # Combine the tokens in the answer and print it out.""
            answers[key] = answer.replace("#","")
            
        # convert words to numbers
        answers['heading'] = "".join(number_map[word] for word in answers['heading'].split())
            
        # cleaning up the output
        answers['tool'] = re.sub(r'\s*-\s*', '-', answers['tool'])
        answers['target'] = re.sub(r'\s+,', ',', answers['target'])
        if answers['tool'] == 'emp':
            answers['tool'] = answers['tool'].upper()
        
        return answers
        # END CODE
        #return {"heading": "", "tool": "", "target": ""}
        