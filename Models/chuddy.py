# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:18:40 2023

@author: debian-os
"""
import torch
import transformers
from transformers import BertTokenizer
import torch import nn
import torch.nn.functional as F
from models.qformer import BertConfig, BertModel, BertLMHeadModel
from models.utils import init_tokenizer

class LayerNorm(nn.LayerNorm):
    def forward(self,x:torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class Chuddy(nn.Module):
    def __init__(self,
                 config = '',
                 image_size = 224,
                 vit = '',
                 embed_dim=256,
                 pixel_values,
                 ):
        super().__init__()
        
        ########===========Qformer Configuration===============########
        encoder_config = BertConfig.from_pretrained('bert-base-uncased')
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attn_freq
        encoder.config.query_length = num_query_tokens
        language_model = LlamaForCausalLM.from_pretrained('')
        
        ########===========submodule initialization==========###########
        self.visual_encoder = vit
        self.query_tokens = nn.Parameter(torch.zeros(1,num_query_tokens,encoder_config.hidden_size))
        self.qformer = BertLMHeadModel(config=encoder_config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.add-special_tokens({"bos_token":"[DEC]"})
        self.language_projection = nn.Linear(encoder_coonfig.hidden_size,config.text_config.hidden_size) 
        self.language_model = language_model
        self.vision_proj = nn.Linear(vision_width,embed_dim)
        self.text_proj = nn.Linear(text_width,embed_dim)
        self.itm_head = nn.Linear(text_width,2)
        
        #####==============Feature Functions===========#####
    def get_text_features(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            labels,
            output_attentions,
            output_hidden_states,
            return_dict):
        output_attentions = output_attentions
        output_hidden_states = output_hidden_states
        return_dict = return_dict
        text_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        return text_outputs
        
    def get_image_features(
            self,
            pixel_values,
            output_attentions,
            output_hidden_states,
            return_dict):
        output_attentions = output_attentions
        output_hidden_states = output_hidden_states
        return_dict = return_dict
        vision_outputs = self.visual_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )
        return vision_outputs
    
    
    def qformer_features(
            self,
            input_ids: Optional[torch.FloatTensor]=None,
            pixel_values: Optional[torch.FloatTensor]=None,
            attention_mask=None
            output_attentions: Optional[bool]=None,
            output_hidden_states: Optional[bool]=None,
            return_dict: Optional[bool]=None):
        output_attentions = output_attentions
        output_hidden_states = output_hidden_states
        return_dict = return_dict
        vision_outputs = self.visual_encoder(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        image_embeds = vision_output[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1],
                                          dtype=torch.long,
                                          device=image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0],-1,-1)
        query_outputs = self.qformer(
            input_ids=None,
            attention_mask=attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_attentions,
            return_dict=return_dict)
        return query_outputs
        
    
    #####=================Model's Forward Method ==============##########
    
    def forward(self,image,caption):
        
        text_input = self.tokenizer(caption,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=35,
                                    return_tensors='pt').to(image.device)
        text_features = self.qformer_features(input_ids=text_input.input_ids,
                                           pixel_values, 
                                           output_attentions, 
                                           output_hidden_states, 
                                           return_dict)
        text_embeds = F.normalize(self.text_proj(text_features.last_hidden_state[:,0,:]),dim=-1)
     
        image_features = self.get_image_features(image)
        #image_atts = torch.ones(image_embed.size()[:-1],dtype=torch.long).to(image.device)
        image_embeds = F.normalize(self.vision_proj(image_features[:,0,:]),dim=-1)
       
        #######=============Image-Text-Contrastive==============#############
        
        sim_i2t = (image_embeds @ text_embeds.t()) / self.temp
        sim_t2i = (text_embeds @ image_embeds.t()) / self.temp
        image_sim = image_embeds @ image_embeds.T
        text_sim = text_embeds @ text_embeds.T
        targets = F.softmax(
            (image_sim + text_sim) / 2 * self.temp,dim=-1)
        text_loss = nn.CrossEntropyLoss()(sim_t2i,targets).mean()
        image_loss = nn.CrossEntropyLoss()(sim_i2t,targets.T).mean()
        loss_itc = (image_loss + text_loss) /2.0
        
        #######================Image-Text-Matching==============#######
        #Adapted from salesforce blip_pretrain image-text matching
        bs = image.size(0)
        output_pos = get_qformer_features(
            input_ids=text_input.input_ids,
            image,
            attention_mask=text_input.attention_mask,
            output_attentions,
            output_hidden_states,
            return_dict,
            )
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4
            weights_t2i.fill_diagonal_(0)
            weights_i2t = F.sformax(sim_i2t[:,:bs],dim=1)+1e-4
            weights_i2t.fill_diagonal_(0)
        
        #select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b],1).item()
            image_embed_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)
        #select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b],1).item()
            text_ids_neg.append(text_input.input_ids[neg_idx])
            text_atts_neg.append(text_input.input_ids.attention_mask[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg,dim=0)
        text_atts_neg = torch.stack(text_atts_neg,dim=0)
        text_ids_all = torch.cat([text_input.input_ids,text_ids_neg],dim=0)
        text_atts_all = torch.cat([text_input.attention_mask,text_atts_neg],dim=0)
        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)
        output_neg = self.get_qformer_features(
            text_ids_all,
            attention_mask = text_atts_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict =True
            )
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:],
                                   output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),
                                torch.zeros(2*bs,dtype=torch.long)],dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output,itm_labels)
        
        
        #######===========Language Modelling==============#######
        decoder_input_ids = text_input.input_ids.clone()
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids,-100)
        query_output = self.get_qformer_features(image)[0]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],dtype=torch.long,device=language_model_inputs.device
            )
        input_embeds = self.language_model.get_input_embeddings()(decoder_input_ids)
        input_embeds = torch.cat([language_model_inputs,inputs_embeds],dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat([language_model_attention_mask,
                                    attention_mask.to(expected_device)],dim=1)
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)
        logits = outputs.logits if return_dict else outputs[0]
        labels = decoder_targets
        labels = labels.to(logits.device)
        logits = logits[...,-labels.size(1):,:]
        shift_logits = logits[...,:-1,:].contiguous()
        shift_labels = labels[...,1:].contiguous().to(logits.device)
        loss_fct = F.cross_entropy(reduction='mean')
        loss_lm = loss_fct(shift_logits.view(-1,self.config.text-config.vocab_size),shift_labels.view(-1))
        return loss_itc,loss_itm,loss_lm
    
               

