from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import spacy
from googletrans import Translator
from summarizer import Summarizer
from transformers import *
import pandas as pd
import re
import string
from googletrans import Translator
from typing import List, Dict, Any


class TextSummarizer():
    
    def __init__(self, 
                 texts: List[str], 
                 context: str = "science" or "general"):
        '''
        Arguments:
            texts: list of different paragraphs which need to be summarized
            context: context of texts, if it's science-based such as biology, chemistry, and physics, ect,
                     set the value as "science", "general" otherwise
        '''
        self._texts = self.clean(texts)
        self._context = context
        
    
    def clean(self,texts: List[str]) -> List[str]:
        '''
        Delete Non-ASCII character & line breaker
        '''
        clean_texts = []
        for text in texts:
            valid_characters = string.printable
            text = text.replace('\n',' ')
            end_string = ''.join(i for i in text if i in valid_characters)
            clean_texts.append(end_string)
            
        return clean_texts
    
    def summarize(self, text:str) -> str:
        '''
        Summarize single text unit
        '''
        
        if self._context  == 'general':
            model = Summarizer()
            opt_k = model.calculate_optimal_k(text)
            summ = model(text, num_sentences=opt_k)
            return summ
        
        elif self._context  == 'science':
            custom_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased')
            custom_config.output_hidden_states=True
            custom_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            custom_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', config=custom_config)
            model = Summarizer(custom_model=custom_model, custom_tokenizer=custom_tokenizer)
            summ = model(text)
            return summ
    
    @staticmethod
    def translate_(original_text: str, src_lan: str = 'en', dest_lan: str = 'fr') -> str:
        '''
        Translation 
        '''
        translator = Translator()
        tran = translator.translate(original_text, src = src_lan, dest = dest_lan).text
        return tran
        
    
        
    def format_results(self,texts: List[Dict]) -> List[Dict]:
        '''
        Format summarizaions along with original texts into seperate dictionaries
        being put into a list
        '''
        results = []
        for i in range(len(texts)):
            text = texts[i]
            summ = self.summarize(text)
            results.append({'original texts': text,
                            'summarization': summ,
                           })
            
        return results
        