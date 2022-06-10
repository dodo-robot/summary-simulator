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
   
    def summarize(self, txt: list):
        final_summary = ""
        data = self.tokenizer(txt, max_length=4096, return_tensors='pt', 
                              padding="max_length", truncation=True).to(self.device)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        sample_outputs = self.model.generate(input_ids,
                                            do_sample=True, 
                                            min_length=0,
                                            max_length=200,
                                            top_k=50, 
                                            top_p=0.95, 
                                            length_penalty=float(2),
                                            repetition_penalty=2.0,
                                            num_return_sequences=1,
                                            )
        
        print("Output:\n" + 100 * '-')
        for i, sample_output in enumerate(sample_outputs):
            summary = self.tokenizer.decode(sample_output, skip_special_tokens=True)
            final_summary += summary+"\n"
            print("{}: {}".format(i, summary))

        return final_summary 












