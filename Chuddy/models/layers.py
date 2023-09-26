# adapted from NExT-GPT github repository
''' layers used in mapping text embedding to visual outputs'''

import torch
from torch import nn
from Chuddy.models.qformer import BertLMHeadModel, BertConfig

class TextFcLayer(nn.Module):
  @classmethod
  def init_qformer(cls,num_query_token,vision_width,num_hidden_layers=2,cross_attention_freq=1):
    encoder_config = BertConfig.from_pretrained('bert-base-uncased')
    encoder_config.encoder_width = vision_width
    encoder_config.hidden_layers = num_hidden_layers
    encoder_config.add_cross_attention = True
    encoder_config.cross_attention_freq = cross_attention_freq
    encoder_config.query_length = num_query_tokens
    Qformer = BertLMHeadModel.from_pretrained('bert-base-uncased',config=encoder_config)
    query_tokens = nn.Parameter(torch.zeros(1,num_query_tokens,encoder_config.hiddeen_size))
    query_tokens.data.normal(mean=0.0,std=encoder_config.initialize_range)
    return Qformer, query_tokens
  
  def __init__(self,
               in_dim: int,
               out_dim: int,
               num_input_tokens: int=1,
               num_output_tokens: int =1,
               mode: str='linear',
               freeze_qformer=False):
                 super().__init__()
                 self.num_input_tokens = num_input_tokens
                 self.num_output_tokens = num_output_tokens
                 self.mode = mode
                 self.out_dim = out_dim
                 if mode == 'linear':
                   self.model = nn.Linear(in_dim,out_dim)
                 elif mode == 'transformer':
                   hidden_dim=512
                   self.fc = nn.Linear(in_dim,hidden_dim)
                   self.tfm = nn.Transformer(batch_first=True,norm_first=True,
                                             d_model=hidden_dim,
                                             num_encoder_layers=4,
                                             num_decoder_layers=4,
                                             dim_feedforward=hidden_dim * 4,
                                             dropout=0.0,
                                             nhead=4)
                   self.model = nn.Linear(hidden_dim,out_dim)
                   self.query_embs = nn.Patameter(torch.randn(1,num_output_tokens,hidden_dim))
                 elif mode == 'qformer':
                  
                   hidden_dim = 768
                   self.fc = nn.Linear(in_dim,hidden_dim)
                   self.Qformer, self.query_tokens = self.init_qformer(num_output_tokens,hidden_dim)
                   self.Qformer.cls = None
                   self.Qformer.bert.embeddings.word_embeddings = None
                   self.Qformer.bert.embeddings.position_embeddings = None
                   for layer in self.Qformer.bert.encoder.layer:
                     layer.output = None
                     layer.intermediate = None
                   self.model = nn.Linear(hidden_dim,out_dim)
                   print('done loading qformer')
                 else:
                   raise NotImplementedError(mode)
  
  def forward(self,
              x: torch.Tensor,
              input_embeds: torch.Tensor) -> torch.Tensor:
                outputs = None
                if isinstance(self.model,nn.ModuleList):
                  assert len(self.model) == x.shape[1] == self.num_input_tokens, (len(self.model), x.shape, self.num_input_tokens)
                  outputs = []
                  for i in range(self,num_input_tokens):
                    outputs.append(self.model[i](x[:,i,:]))
                  outputs = torch.stack(outputs,dim=1)
                elif self.mode == 'transformer':
                  x = x + input_embeds
                  x = self.fc(X)
                  X = self.tfm(x,self.qurey_embs.repeat(x.shape[0],1,1))
                  outputs = self.model(x)
                  if outputs.shape[1] != self.num_output_tokens and self.mode == 'linear':
                    if self.mode == 'linear':
                      outputs = outputs[:,:self.num_output_tokens,:]
                    else:
                      raise NotImplementedError
                elif self.mode == 'qformer':
                  x = x + input_embeds
                  x = self.fc(x)
                  image_atts = torch.ones(x.size()[:-1],dtype=torch.long).to(x.device)
                  query_tokens = self.query_tokens.expand(x.shape[0],-1,-1)
                  outputs = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=x,
                    encoder_attention_mask=image_atts,
                    return_dict=True,).last_hidden_state
                  outputs = self.model(outputs)
                assert outputs.shape[1] == 1 or (outputs.shape[1] * outputs.shape[2] == self.num_output_ * self.out_dim),(
                  outputs.shape, self.num_output_tokens)
                return outputs
                    
                   















