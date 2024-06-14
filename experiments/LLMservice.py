import requests
import ollama
class LLM_serv:
    def __init__(self,url="http://tentris-ml.cs.upb.de:8000",model="llama3:70b"):
        self.url=url
        self.model=model


    def run_ner_request(self, sentence:str):
        print(requests.get(url=self.url).json()["response"])
        response = requests.get(url=self.url + "/api/generate",
                                headers={"accept": "application/json", "Content-Type": "application/json"},
                                json={"model": self.model, "prompt":"Please extract all entities from the following sentence and return a list in JSON format using the key \"entitiy\": "+ sentence+" Please do not format the json output"})
        return response.json()["response"]

    def run_expansion_request(self, sent:str,spans):
        req=f'''
            Given the sentence {sent} and the spans {spans}. Can you map each of the given spans to its likely entity in wikipedia and return a valid json document using the keys "span" and "entity_name" please do not format the json output?
            '''
        print(req)
        print(requests.get(url=self.url).json()["response"])
        response = requests.get(url=self.url + "/api/generate",
                                headers={"accept": "application/json", "Content-Type": "application/json"},
                                json={"model": self.model, "prompt":req})
        return response.json()["response"]

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


ls=Local_service("llama3")

print(ls.run_ner_request("Barbara Walters stands by Rosie O'Donnell ‘View' host denies Trump's claim she wanted comedian off morning show NEW YORK - Barbara Walters is back from vacation — and she's standing by Rosie O'Donnell in her bitter battle of words with Donald Trump. Walters, creator of ABC's 'The View,' said Wednesday on the daytime chat show that she never told Trump she didn't want O'Donnell on the show, as he has claimed. 'Nothing could be further from the truth,' she said. 'She has brought a new vitality to this show and the ratings prove it,' Walters said of O'Donnell, who is on vacation this week. When she returns, Walters said, 'We will all welcome her back with open arms.' Walters also took a moment to smooth things over with The Donald, who got all riled up when O'Donnell said on 'The View' that he had been 'bankrupt so many times.' 'ABC has asked me to say this just to clarify things, and I will quote: ‘Donald Trump has never filed for personal bankruptcy. Several of his casino companies have filed for business bankruptcies. They are out of bankruptcy now,'' Walters said. O'Donnell and Trump have been feuding since he announced last month that Miss USA Tara Conner, whose title had been in jeopardy because of underage drinking, would keep her crown. Trump is the owner of the Miss Universe Organization, which includes Miss USA and Miss Teen USA. The 44-year-old outspoken moderator of 'The View,' who joined the show in September, said Trump's news conference with Conner had annoyed her 'on a multitude of levels' and that the twice-divorced real estate mogul had no right to be 'the moral compass for 20-year-olds in America.' Trump fired back, calling O'Donnell a 'loser' and a 'bully,' among other insults, in various media interviews. He is the host of NBC's 'The Apprentice").replace("\n",""))
