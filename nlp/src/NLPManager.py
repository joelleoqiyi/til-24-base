from typing import Dict
# ADDED:
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
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

questions = {
    'heading': "What is the heading value?", 
    'target': "What is the full target description?",
    'tool': "What is the tool to deploy?"
}

class NLPManager:
    def __init__(self):
        # initialize the model here
        # START CODE
        model = torch.load('model_nlp.pth')
        tokenizer = torch.load('tokenizer_nlp.pth')
        # device = 0 if torch.cuda.is_available() else -1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipe = pipeline('question-answering', device=device, model=model, tokenizer=tokenizer)
        # END CODE

    def qa(self, context: str) -> Dict[str, str]:
        # perform NLP question-answering
        # START CODE
        answers = {}
        
        for key, question in questions.items():
            output = self.pipe(question=question, context=context)
            answers[key] = output['answer']
            # output example = {'score': 0.956532895565033, 'start': 90, 'end': 101, 'answer': 'machine gun'} 
            
        # post-processing
        try:
            answers['heading'] = "".join(number_map[word] for word in answers['heading'].split())
        except:
            pass
        answers['tool'] = re.sub(r'\s*-\s*', '-', answers['tool'])
        answers['target'] = re.sub(r'\s+,', ',', answers['target'])
        
        return answers
        # END CODE
        #return {"heading": "", "tool": "", "target": ""}