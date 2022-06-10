# Class to call for generating summaries

from transformers import AutoTokenizer
from simulator.summarizer.longformer_encoder_decoder import LongformerEncoderDecoderForConditionalGeneration
from typing import List
import torch

class BARTSummarizer():
    
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LongformerEncoderDecoderForConditionalGeneration.\
            from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.model.model.encoder.config.gradient_checkpointing = True
        self.model.model.decoder.config.gradient_checkpointing = True
        print('Successfully started the model')
        
    
    # {
    #   temperature: 0.85, 
    #   num_beams: int(3),
    #   length_penalty: float(2), 
    #   max_length: int(200), 
    #   min_length: int(0), 
    #   no_repeat_ngram_size: int(3)  
    # }
    def beam_summary(self, input_ids, **args):
        return self.model.generate(input_ids, do_sample = False, 
                                   temperature=args['temperature'], num_beams=args['num_beams'],
                                   length_penalty=args['length_penalty'], max_length = args['max_length'],
                                   min_length=args['min_length'], no_repeat_ngram_size=args['no_repeat_ngram_size'])

        
    # {
    # min_length: 0,
    # max_length: 200,
    # top_k: 50, 
    # top_p: 0.95, 
    # length_penalty: float(2),
    # repetition_penalty: 2.0,
    # num_return_sequences: 1,
    # }
    def sample_summary(self, input_ids, **args):
        return  self.model.generate(input_ids,
                            do_sample=True, 
                            min_length=args['min_length'],
                            max_length=args['max_length'],
                            top_k=args['top_k'],
                            top_p=args['top_p'],
                            length_penalty=args['length_penalty'],
                            repetition_penalty=args['repetition_penalty'],
                            num_return_sequences=args['num_return_sequences'],
                            )
        
    def summarize(self, txt: list, **args):
        final_summary = ""
        data = self.tokenizer(txt, max_length=4096, return_tensors='pt', 
                              padding="max_length", truncation=True).to(self.device)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        if args['tipo']=='sample':
            sample_outputs = self.sample_summary(input_ids, args)
        else:
            sample_outputs = self.beam_summary(input_ids, args)
        
        for i, sample_output in enumerate(sample_outputs):
            summary = self.tokenizer.decode(sample_output, skip_special_tokens=True)
            final_summary += summary+"\n"

        return final_summary 












