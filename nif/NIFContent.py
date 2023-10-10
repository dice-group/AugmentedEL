# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:13:32 2018

@author: Daniel
"""
#,**kwargs
#from rdflib import Literal

class NIFContent:
    def __init__(self,uri):
        self.uri=uri
        self.endStatement=None
        self.a='nif:RFC5147String , nif:String, nif:Context'
        self.begin_index=None
        self.end_index=None
        self.is_string=None
        self.anchor_of=None
        self.reference_context=None
        self.taIdentRef=None
        self.taClassRef=None
        
    
    def get_NIF_String(self):
        string=''
        string+='<'+self.uri+'>\n'
        string+='        a                     '+self.a+' ;\n'
        string+='        nif:beginIndex        "'+str(self.begin_index)+'"^^xsd:nonNegativeInteger ;\n'       
        string+='        nif:endIndex          "'+str(self.end_index)+'"^^xsd:nonNegativeInteger '
        if self.reference_context is not None:
            string+=';\n        nif:referenceContext  <'+str(self.reference_context)+'> '
        if not self.anchor_of is None:
            string+=';\n        nif:anchorOf          "'+self.anchor_of.replace('"',"'").replace('%',"P")+'" '
        if not self.is_string is None:
            string+=';\n        nif:isString          "'+str(self.is_string.replace('"',"'").replace('%',"P"))+'"^^xsd:string '
        if self.taIdentRef is not None:
            string+=';\n        itsrdf:taIdentRef     <'+str(self.taIdentRef)+'> '
        if self.taClassRef is not None:
            string+=';\n        itsrdf:taClassRef     <'+str(self.taClassRef)+'> '
        string+='.\n\n'
        return string
        
    def set_begin_index(self,string):
        self.begin_index=string
    def set_end_index(self,string):
        self.end_index=string
    
    def set_is_string(self,string):
        self.is_string=string
    
    def set_anchor_of(self,string):
        self.anchor_of=string
    
    def set_reference_context(self,string):
        self.reference_context=string
    
    def set_taIdentRef(self,string):
        self.taIdentRef=string
        
    def set_taClassRef(self,string):
        self.taClassRef=string
    def set_end_statement(self,string):
        self.endStatement=string