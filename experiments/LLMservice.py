import requests
import ollama


class Local_service:
    def __init__(self,model="llama3:70b"):
        self.model=model
    def run_expansion_request(self,sent:str,spans):
        req = f'''
                    Given the sentence {sent} and the spans {spans}. Can you map each of the given spans to its likely entity in wikipedia and return a valid json document using the keys "span" and "entity_name" please do not format the json output?
                '''
        response = ollama.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': req,
            },
        ])
        return response['message']['content']
    def run_ner_request(self,sentence:str):
        req="Please generate one list with all entities from the following text in JSON format, excluding numbers. Do not format the json output."+ sentence
        response = ollama.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': req,
            },
        ])
        return response['message']['content']
    def expand_ner_request(self,sentence:str):
        req="In the following text there are already annotated entities, if the text contains more entities, please provide a list with only those entities in json format. Do not format the JSON output. "+ sentence
        response = ollama.chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': req,
            },
        ])
        return response['message']['content']


