# Model configuration class for the Visual encoder, qformer and Language model

import torch
from torch import nn
class ModelConfig(nn.Module):
  def __init__(self,
               bert_config = 'bert-base-uncased',
               llama_config = 'meta-llama/Llama-2-13b-chat-hf',
               beit_config = 'microsoft/beit-base-patch16-224-pt22k'
              ):
                super().__init__()
                self.bert_config = bert_config
                self.llama_config = llama_config
                self.beit_config = beit_config
                
