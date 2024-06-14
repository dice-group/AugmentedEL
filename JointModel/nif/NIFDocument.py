# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:50:34 2018

@author: Daniel
"""
import re
from nif.NIFContent import NIFContent

class NIFDocument:
    def __init__(self):
        self.pref_rdf="@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."
        self.pref_xsd="@prefix xsd: <http://www.w3.org/2001/XMLSchema#> ."
        self.pref_nif="@prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#> ."
        self.pref_itsrdf="@prefix itsrdf: <http://www.w3.org/2005/11/its/rdf#> ."
        self.nifContent=[]
    
    
    def addContent(self, cont):
        self.nifContent.append(cont)
    #return nif docuemt as a string
    def get_nif_string(self):
        string=""
        string+=self.pref_rdf+'\n'
        string+=self.pref_xsd+'\n'
        string+=self.pref_nif+'\n'
        string+=self.pref_itsrdf+'\n'
        string+=self.pref_rdf+'\n'
        string+='\n'
        for nifCont in self.nifContent:
            string+=nifCont.get_NIF_String()
        return string
   


    #get NIF Cotnet with full text
    def get_referenced_contex_id(self):
        i=0
        while i<len(self.nifContent):
            if not self.nifContent[i].is_string is None:
                return i
            i=i+1
        return None
#transform String in nif document
def nifStringToNifDocument(NIFString):
    nifStatements=NIFString.split("\n")
    document=NIFDocument()
    nifStatements=list(filter(None,nifStatements))
    i=0
    while i<len(nifStatements):
        if nifStatements[i].startswith('<http'):
            uri=str(nifStatements[i])
            nifContent=NIFContent(uri[1:len(uri)-1])
            contComplete=False
            i=i+1
            while not contComplete:
                if re.search('nif:beginIndex[ \t]+"([0-9]+)"\^\^xsd:nonNegativeInteger',nifStatements[i]):
                    m=re.search('nif:beginIndex[ \t]+"([0-9]+)"\^\^xsd:nonNegativeInteger',nifStatements[i])
                    nifContent.set_begin_index(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                if re.search('nif:endIndex[ \t]+"([0-9]+)"\^\^xsd:nonNegativeInteger',nifStatements[i]):
                    m=re.search('nif:endIndex[ \t]+"([0-9]+)"\^\^xsd:nonNegativeInteger',nifStatements[i])
                    nifContent.set_end_index(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                if re.search('nif:isString[ \t]+"(.*)"\^\^xsd:string',nifStatements[i]):
                    m=re.search('nif:isString[ \t]+"(.*)"\^\^xsd:string',nifStatements[i])
                    nifContent.set_is_string(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                if re.search('nif:isString[ \t]+"(.*)" (;|.)',nifStatements[i]):
                    m=re.search('nif:isString[ \t]+"(.*)" (;|.)',nifStatements[i])
                    nifContent.set_is_string(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                if re.search('nif:anchorOf[ \t]+"(.*)"',nifStatements[i]):
                    m=re.search('nif:anchorOf[ \t]+"(.*)"',nifStatements[i])
                    nifContent.set_anchor_of(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                if re.search('nif:referenceContext[ \t]+<(https?://.*)>',nifStatements[i]):
                    m=re.search('nif:referenceContext[ \t]+<(https?://.*)>',nifStatements[i])
                    nifContent.set_reference_context(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True                
                if re.search('itsrdf:taIdentRef[ \t]+<(https?://.*)>',nifStatements[i]):
                    m=re.search('itsrdf:taIdentRef[ \t]+<(https?://.*)>',nifStatements[i])
                    nifContent.set_taIdentRef(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                if re.search('itsrdf:taClassRef[ \t]+<(https?://.*)>',nifStatements[i]):
                    m=re.search('itsrdf:taClassRef[ \t]+<(https?://.*)>',nifStatements[i])
                    nifContent.set_taClassRef(m.group(1))
                    if nifStatements[i].endswith('.'):
                        contComplete=True
                        nifContent.set_end_statement(m.group(1))
                if contComplete:
                    document.addContent(nifContent)
                else:
                    i=i+1
        i=i+1
    return document
                

    
    
    
        
        