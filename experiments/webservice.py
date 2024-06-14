# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:40:19 2018

@author: Daniel
"""

from flask import Flask
from flask import request
from flask import make_response
from flask import render_template
from flask_restful import Resource, Api
#from rasa_nlu.model import Metadata, Interpreter
import requests
import configparser
from eval_script import EL_model
from nif import NIFDocument as NIFDocument
from nif import NIFContent as NIFContent
from nif_api import lm_output_to_annoation_ner, lm_output_to_annoation_e2e
#from GENRE.genre_md import GENREAnnotator

import pickle
app = Flask(__name__)
api = Api(app)
el_model_e2e=EL_model("../joint_model/e2e_aida",200)
el_model=EL_model("../joint_model/aida-125ep",200,apply_llm_ner=False)

ge_model=None
from flair.data import Sentence
from flair.nn import Classifier
tagger = Classifier.load('ner')
prefix_allowed_token_fn_el=el_model.get_pref_allowed_token_fn()
prefix_allowed_token_fn_ner=el_model.get_pref_allowed_token_fn()
#config = configparser.RawConfigParser()
#config.read('conf/conf.cnf')

#rasa_conf = config.get('rasa', 'rasa_config_file')
#agdistis_url = config.get('agdistis', 'agdistis_url')
#interpreter = Interpreter.load(config.get('rasa', 'rasa_model'), rasa_conf)
#host = config.get('flask', 'host')
wikidata_uris=pickle.load(open("../data/wikidata_uris.pkl","rb"))
print("finished loading")

def add_nif_entities(reference_context, base_uri, entities, doc):
    for ent in entities:
        nif_content = NIFContent.NIFContent(base_uri + '#' + str(ent['start']) + ',' + str(ent['end']))
        nif_content.set_begin_index(ent['start'])
        nif_content.set_end_index(ent['end'])
        nif_content.set_reference_context(reference_context)
        nif_content.set_anchor_of(ent['value'])
        doc.addContent(nif_content) 
    return doc


# webservice for entity recognition
@app.route('/ner/', methods=['POST'])
def annotate_nif_string():
    string = request.data.decode()
    print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    lm_output=el_model.predict_ner(doc.nifContent[0].is_string)
    print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_ner(lm_output,doc.nifContent[0].uri)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp



@app.route('/e2e_direct/', methods=['POST'])
def annotate_nif_string_e2e_direct():
    string = request.data.decode()
    #print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    #prefix_allowed_token_fn_ner=EL_model.get_prefix_allowed_token_fn_ner("Text to annotate:"+doc.nifContent[0].is_string)
    lm_output=el_model_e2e.predict_el_e2e(doc.nifContent[0].is_string)
    #print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_e2e(lm_output,doc.nifContent[0].uri,wikidata_uris)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp


@app.route('/e2e_direct_exp/', methods=['POST'])
def annotate_nif_string_e2e_direct_exp():
    string = request.data.decode()
    #print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    #prefix_allowed_token_fn_ner=EL_model.get_prefix_allowed_token_fn_ner("Text to annotate:"+doc.nifContent[0].is_string)
    lm_output=el_model_e2e.predict_el_e2e_exp(doc.nifContent[0].is_string)
    #print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_e2e(lm_output,doc.nifContent[0].uri,wikidata_uris)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp


@app.route('/e2e/', methods=['POST'])
def annotate_nif_string_e2e():
    string = request.data.decode()
    #print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    #prefix_allowed_token_fn_ner=EL_model.get_prefix_allowed_token_fn_ner("Text to annotate:"+doc.nifContent[0].is_string)
    lm_output=el_model.predict_e2e(doc.nifContent[0].is_string,)
    #print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_e2e(lm_output,doc.nifContent[0].uri,wikidata_uris)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp

@app.route('/e2e_mixed/', methods=['POST'])
def annotate_nif_string_e2e_e2e():
    string = request.data.decode()
    #print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    #prefix_allowed_token_fn_ner=EL_model.get_prefix_allowed_token_fn_ner("Text to annotate:"+doc.nifContent[0].is_string)
    lm_output=el_model.predict_mixed_e2e(doc.nifContent[0].is_string,el_model_e2e,False)
    #print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_e2e(lm_output,doc.nifContent[0].uri,wikidata_uris)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp



@app.route('/e2e-flair/', methods=['POST'])
def e2e_flair():
    string = request.data.decode()
    # print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    print(doc.nifContent[0].is_string)
    text=doc.nifContent[0].is_string
    sentence = Sentence(text)
    tagger.predict(sentence)
    # lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    curr_ind = 0
    # print the sentence with all annotations
    for sp in sentence.annotation_layers["ner"]:
        text_to_annotate = text[curr_ind:]
        text_to_annotate = text_to_annotate.replace(sp.data_point.text,
                                                    "[START_ENT] " + sp.data_point.text + " [END_ENT]", 1)
        text = text[0:curr_ind] + text_to_annotate
        curr_ind = text.rindex("[END_ENT]") + len(["END_ENT"])
    print(text)
    lm_output=el_model.predict_disambiguation_only(text,False)
    print(lm_output)
    annotations = lm_output_to_annoation_e2e(lm_output, doc.nifContent[0].uri, wikidata_uris)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    # res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    # app.logger.debug(res)

    # doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp


@app.route('/e2e-flair_ner_exp/', methods=['POST'])
def e2e_flair_ner_exp():
    string = request.data.decode()
    # print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    print(doc.nifContent[0].is_string)
    text=doc.nifContent[0].is_string
    sentence = Sentence(text)
    tagger.predict(sentence)
    # lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    curr_ind = 0
    # print the sentence with all annotations
    for sp in sentence.annotation_layers["ner"]:
        text_to_annotate = text[curr_ind:]
        text_to_annotate = text_to_annotate.replace(sp.data_point.text,
                                                    "[START_ENT] " + sp.data_point.text + " [END_ENT]", 1)
        text = text[0:curr_ind] + text_to_annotate
        curr_ind = text.rindex("[END_ENT]") + len(["END_ENT"])
    print(text)
    lm_output=el_model.predict_disambiguation_flair_with_ner_expansion(text,doc.nifContent[0].is_string)
    print(lm_output)
    annotations = lm_output_to_annoation_e2e(lm_output, doc.nifContent[0].uri, wikidata_uris)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    # res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    # app.logger.debug(res)

    # doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp

@app.route('/e2e-genre/', methods=['POST'])
def annotate_nif_string_e2e_genre():
    string = request.data.decode()
    #print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    annotations=ge_model.predict(doc.nifContent[0].is_string,doc.nifContent[0].uri)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp

@app.route('/e2econst/', methods=['POST'])
def annotate_nif_string_e2e_conste2 ():
    string = request.data.decode()
    #print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    lm_output=EL_model.predict_e2e(doc.nifContent[0].is_string,prefix_allowed_token_fn)
    #print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_e2e(lm_output,doc.nifContent[0].uri,titles_to_wikipedia)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    #app.logger.debug(doc.get_nif_string())
    resp = make_response(doc.get_nif_string())
    resp.headers['content'] = 'application/x-turtle'

    return resp


docs=[]
@app.route('/get_ds/', methods=['POST'])
def annotate_nif_string_extract():
    string = request.data.decode()
    print(string)
    doc = NIFDocument.nifStringToNifDocument(string)
    docs.extend(doc.nifContent)
    #lm_output=EL_model.predict_ner(doc.nifContent[0].is_string)
    docnew=NIFDocument.NIFDocument()
    docnew.nifContent=docs
    with open("extracted_der","w",encoding="utf-8") as out_file:
        out_file.write(docnew.get_nif_string())
    '''
    lm_output=EL_model.predict_e2e(doc.nifContent[0].is_string)
    print("str_comp")
    print(doc.nifContent[0].is_string)
    print(lm_output)
    annotations=lm_output_to_annoation_e2e(lm_output,doc.nifContent[0].uri,titles_to_wikipedia)
    doc.nifContent.extend(annotations)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(doc.nifContent[0].is_string.replace(',', ' ').replace('"', ' ').replace('\\', ''))
    #app.logger.debug(res)

    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    app.logger.debug(doc.get_nif_string())
    '''
    resp = make_response(string)
    resp.headers['content'] = 'application/x-turtle'
    return resp


'''
# webservice for recognition and liking
@app.route('/nifa2kb/', methods=['POST'])
def annotate_nif_string_Linking():
    string = request.data.decode()
    doc = NIFDocument.nifStringToNifDocument(string)
    app.logger.debug(doc.nifContent[int(doc.get_referenced_contex_id())].is_string)
    # replace of comma with whitespace for tokenizer
    #res = interpreter.parse(
    #    doc.nifContent[int(doc.get_referenced_contex_id())].is_string.replace(',', ' ').replace('"', ' ').replace('\\',
                                                                                                                  ''))
    #base_uri = doc.nifContent[0].uri[0:doc.nifContent[0].uri.find('#')]
    #app.logger.debug(res)
    #doc = add_nif_entities(doc.nifContent[0].uri, base_uri, res['entities'], doc)
    # doc.nifContent[int(doc.get_referenced_contex_id())].is_string=doc.nifContent[int(doc.get_referenced_contex_id())].is_string.replace('\\','\\\\')
    #ag_string = doc.get_nif_string()
    #app.logger.debug(ag_string)
    # get links from agdsitis
    #resp = requests.post(agdistis_url, ag_string.encode('utf-8'))
    #app.logger.debug(resp.content.decode())

    resp = make_response(resp.content)
    resp.headers['content'] = 'application/x-turtle'
    app.logger.debug(resp)

    return resp
'''

'''
# tool demo
@app.route('/')
def index():
'''




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)