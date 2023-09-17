# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:18:40 2023

@author: Chima Emmanuel
"""
import torch
import transformers
from typing import Union,Tuple,Optional,Any
from torch import nn
import torch.nn.functional as F
from Chuddy.models.imagebind import *
from Chuddy.models.diffusion_pipelines.sd_pipeline import StableDiffusionPipeline
from Chuddy.models.diffusion_pipelines.ad_pipeline import AudioLDM2Pipeline
from Chuddy.models.diffusion_pipelines.vd_pipeline import TextToVideoSDPipeline
# from Chuddy.models.qformer import BertConfig, BertModel, BertLMHeadModel
from Chuddy.configuration.configuration import  ModelConfig     ## not yet fully implemented
from Chuddy.models.visual_model import BeitModel,BeitConfig
from transformers import LlamaForCausalLM,LlamaTokenizer,LlamaConfig
import logging
# from Chuddy.models.modelling_blip2 import Blip2Base,disabled_train
# from Chuddy.utils.registry import registry
from peft import (
LoraConfig,
get-peft_model,
get_peft_model_state_dict,
prepare_model_for_int8_training,
set_peft_model_state_dict,
)
# Main Model class
class Chuddy(nn.Module):
    def __init__(self,
                 config = ModelConfig(),
                 prompt= "",
                 device_8bit=0,
                 lora_r=0,
                 lora_target_modules=['q_proj','v_proj'],
                 lora_alpha=16,
                 lora_dropout=0.05,
                 **args,
                 ):
        super(Chuddy,self).__init__()
        self.args = args
        ########===========Llama Configuration===============########
        text_config = LlamaConfig.from_pretrained(config.llama_config)
        self.text_config = text_config
        language_model = LlamaForCausalLM.from_pretrained(text_config)
        
        ########===========submodule initialization==========###########
        self.visual_encoder, self.visual_hidden_size = imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        self.lm_tokenizer = LlamaTokenizer.from_pretrained(config.llama_config)
        self.lm_tokenizer.add_special_tokens({'pad_token':'[PAD]'})
        self.lm_tokenizer.add_special_tokens({'bos_token':'</s>'})
        self.lm_tokenizer.add_special_tokens({'eos_token':'</s>'})
        self.lm_tokenizer.add_special_tokens({'unk_token':'</s>'})
        self.tokenizer.add_special_tokens({"bos_token":"[DEC]"})
        self._add_image_token()
        self._add_video_token()
        self._add_audio_token()
        self.language_projection = nn.Linear(self.visual_hidden_size,self.language_model.config.hidden_size) 
        self.language_model = language_model
        self.language_model.resize_token_embeddings(len(self.lm_tokenizer))

        # freeze vision encoder
        for name,param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
            
        logging.info('frozen_vis_encoder enabled')
        if self.args['freeze_lm']:
            for name,param in self.language_model.named_parameters():
                param.requires_grad = False
            self.language_model.eval()
            logging.info('frozen LLM enabled')
        else:
            print('lora tuning Llama')
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha = lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules)
            self.language_model = get_peft_model(self.language_model,peft_config)
            self.language_model.print_trainable_parameters()
        print('LLM initialized')
        if self.args['freeze_input_proj']:
            for param in self.language_projection.parameters():
                params.requires_grad=False
        self.input_embeddings = self.language_model.get_input_embeddings()
                     
        #####==============Feature Functions===========#####
    def _add_image_token(self):
        self.lm_tokenizer.add_tokens({"<Img>"})
        self.lm_tokenizer.add_tokens({"</Img>"})
        # add [img] tokens to vocab
        self.args['gen_img_token_idx']= []
        for i in range(self.args['num_gen_img_tokens']):
            print('adding image tokens to vocab')
            num_added_tokens = self.lm_tokenizer.add_tokens(f'[IMG{i}]')
            gen_token_idx = self.lm_tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_img_token_idx'].append(gen_token_idx[0])
            
    def _add_video_token(self):
        self.lm_tokenizer.add_tokens({"<Vid>"})
        self.lm_tokenizer.add_tokens({"</Vid>"})
        # add [img] tokens to vocab
        self.args['gen_vid_token_idx']= []
        for i in range(self.args['num_gen_vid_tokens']):
            print('adding image tokens to vocab')
            num_added_tokens = self.lm_tokenizer.add_tokens(f'[VID{i}]')
            gen_token_idx = self.lm_tokenizer(f'[VID{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_vid_token_idx'].append(gen_token_idx[0])

    def _add_audio_token(self):
        self.lm_tokenizer.add_tokens({"<Aud>"})
        self.lm_tokenizer.add_tokens({"</Aud>"})
        # add [img] tokens to vocab
        self.args['gen_audio_token_idx']= []
        for i in range(self.args['num_gen_audio_tokens']):
            print('adding image tokens to vocab')
            num_added_tokens = self.lm_tokenizer.add_tokens(f'[AUD{i}]')
            gen_token_idx = self.lm_tokenizer(f'[AUD{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_audio_token_idx'].append(gen_token_idx[0])
            
    def get_text_encoding(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor]=None,
            decoder_input_ids: Optional[torch.Tensor]=None,
            decoder_attention_mask: Optional[torch.Tensor]=None,
            labels: Optional[torch.Tensor]=None,
            output_attentions: Optional[bool]=None,
            output_hidden_states: Optional[bool]=None,
            return_dict: Optional[bool]=None):
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
        
    def get_image_encoding(
            self,
            image_path):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_path,self.device)}
        inputs = {key: inputs[key].to(self.language_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            image_embeds = embeddings['vision']
        inputs_llm = self.language_projection(image_embeds).unsqueeze(1)
        atts_llm = torch.ones(inputs_llm.size()[:-1],dtype=torch.long).to(self.ddevice)
        return inputs_llm,atts_llm

    
    def get_video_encoding(
            self,
            video_path):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(video_path,self.device)}
        inputs = {key: inputs[key].to(self.language_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            video_embeds = embeddings[ModalityType.VISION]
        inputs_llm = self.language_projection(video_embeds).unsqueeze(1)
        atts_llm = torch.ones(inputs_llm.size()[:-1],dtype=torch.long).to(self.ddevice)
        return inputs_llm,atts_llm

    
    def get_audio_encoding(
            self,
            audio_path):
        inputs = {ModalityType.AUDIO: data.load_and_transform_vision_data(audio_path,self.device)}
        inputs = {key: inputs[key].to(self.language_model.dtype) for key in inputs}
        with torch.no_grad():
            embeddings = self.visual_encoder(inputs)
            audio_embeds = embeddings[ModalityType.AUDIO]
        inputs_llm = self.language_projection(audio_embeds).unsqueeze(1)
        atts_llm = torch.ones(inputs_llm.size()[:-1],dtype=torch.long).to(self.ddevice)
        return inputs_llm,atts_llm


    ###==============================Decoder-side-module-alignment=====================########
    # alignment module for LLM-to_image adapted from Next-GPT
    self.sd_ckpt_path = self.args['image_diffusion']
    self.gen_text_hidden_fcs = nn.ModuleList([])
    for layer_idx in self.args['text_emb_to_img_layers']:
        if layer_idx == -1 or layer_idx == self.language_model.config.num_hidden_layers:
            in_dim = self.language_model.config.hidden_size
            self.gen_text_hidden_fcs.append(
                TextFcLayer(in_dim,768,
                            num_input_tokens=self.args['num_gen_img_tokens'].
                            num_output_tokens=self.args['num_clip_tokens'],
                            mode = self.args['text_fc_to_img_mode'])
            )
        elif layer_idx < self.language_model.config.num_hidden_layers:
            self.gen_text_hidden_fcs.append(
              TextFcLayer(self.language_model.config.hidden_size,768,
                            num_input_tokens=self.args['num_gen_img_tokens'].
                            num_output_tokens=self.args['num_clip_tokens'],
                            mode = self.args['text_fc_to_img_mode'])
        )
        else: 
            raise ValueError(f'embedding of layer {layer_idx} was requested but model only has {self.language_model.config.num_hidden_layers}')

    
    # alignment module for LLM-to_video adapted from Next-GPT
    self.sd_ckpt_path = self.args['video_diffusion']
    self.gen_text_hidden_fcs_video = nn.ModuleList([])
    for layer_idx in self.args['text_emb_to_video_layers']:
        if layer_idx == -1 or layer_idx == self.language_model.config.num_hidden_layers:
            in_dim = self.language_model.config.hidden_size
            self.gen_text_hidden_fcs_video.append(
                TextFcLayer(in_dim,1024,
                            num_input_tokens=self.args['num_gen_video_tokens'].
                            num_output_tokens=self.args['num_clip_tokens'],
                            mode = self.args['text_fc_to_video_mode'])
            )
        elif layer_idx < self.language_model.config.num_hidden_layers:
            self.gen_text_hidden_fcs_video.append(
              TextFcLayer(self.language_model.config.hidden_size,1024,
                            num_input_tokens=self.args['num_gen_video_tokens'].
                            num_output_tokens=self.args['num_clip_tokens'],
                            mode = self.args['text_fc_to_video_mode'])
        )
        else: 
            raise ValueError(f'embedding of layer {layer_idx} was requested but model only has {self.language_model.config.num_hidden_layers}')

     
    # alignment module for LLM-to_Audio adapted from Next-GPT
    self.sd_ckpt_path = self.args['audio_diffusion']
    self.gen_text_hidden_fcs_audio = nn.ModuleList([])
    for layer_idx in self.args['text_emb_to_audio_layers']:
        if layer_idx == -1 or layer_idx == self.language_model.config.num_hidden_layers:
            in_dim = self.language_model.config.hidden_size
            self.gen_text_hidden_fcs_audio.append(
                TextFcLayer(in_dim,512,
                            num_input_tokens=self.args['num_gen_audio_tokens'].
                            num_output_tokens=self.args['num_clip_tokens'],
                            mode = self.args['text_fc_to_audio_mode'])
            )
        elif layer_idx < self.language_model.config.num_hidden_layers:
            self.gen_text_hidden_fcs_audio.append(
              TextFcLayer(self.language_model.config.hidden_size,512,
                            num_input_tokens=self.args['num_gen_audio_tokens'].
                            num_output_tokens=self.args['num_clip_tokens'],
                            mode = self.args['text_fc_to_audio_mode'])
        )
        else: 
            raise ValueError(f'embedding of layer {layer_idx} was requested but model only has {self.language_model.config.num_hidden_layers}')
    

    if self.args['freeze_output_proj']:
        for name,params in self.gen_text_hidden_fcs.named_parameters():
            params.requires_grad = False
        for name,params in self.gen_text_hidden_fcs_video.named_parameters():
            params.requires_grad = False
        for name,params in self.gen_text_hidden_fcs_audio.named_parameters():
            params.requires_grad = False
    #####=================Model's Forward Method ==============##########
    
    def forward(self,
                image,
                caption,
                output_attentions: Optional[bool]=None,
                output_hidden_states: Optional[bool]=None,
                return_dict: Optional[bool]=None):
        
        text_input = self.tokenizer(caption,
                                    padding='max_length',
                                    truncation=True,
                                    max_length=35,
                                    return_tensors='pt').to(image.device)
        text_features = self.qformer_features(input_ids=text_input.input_ids,
                                           output_attentions=output_attentions, 
                                           output_hidden_states=output_hidden_states, 
                                           return_dict=return_dict)
        text_embeds = F.normalize(self.text_proj(text_features.last_hidden_state[:,0,:]),dim=-1)
     
        image_features = self.qformer_features(pixel_values=image,
                                               return_dict=return_dict)
                                               
        #image_atts = torch.ones(image_embed.size()[:-1],dtype=torch.long).to(image.device)
        image_embeds = F.normalize(self.vision_proj(image_features.last_hidden_state),dim=-1)
                    
        #######===========Language Modelling==============#######
        self.prompt = prompt
        self.prompt_tokens = self.lm_tokenizer(self.prompt)
        self.lm_tokenizer.padding_side='right'
        self.lm_tokenizer.truncation_side='left'
        decoder_input_ids = self.lm_tokenizer(caption,
                                             return_tensors='pt',
                                             truncation=True,
                                             max_length=self.max_text_length)
        decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids,-100)
        query_output = self.qformer_features(image)[0]
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
                                    decoder_input_ids.attention_mask.to(expected_device)],dim=1)
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
        loss_hyp = outputs.loss
        loss_lm = loss_fct(shift_logits.view(-1,self.text_config.vocab_size),shift_labels.view(-1))
        return loss_itc,loss_itm,loss_lm,loss_hyp
    ######==============model's generate function===========########
    @torch.no_grad()
    def generate(self,
                 image: Optional[torch.FloatTensor]=None,
                 prompt: Optional[torch.LongTensor]=None,
                 max_length=256,
                 min_length=1,
                 temperature=1,
                 attention_mask: Optional[torch.LongTensor]=None,
                 **generate_kwargs,
                ):
        self.lm_tokenizer.padding_size = 'left'
        if image is not None:
            prompt = self.prompt
            image = image
            bs = image.size(0)
            query_tokens = self.query_tokens.expand(bs,-1,-1)
           # text_qformer = self.tokenizer(
            image_embeds = get_image_features(image,return_dict=True)
            image_attention_mask = torch.ones(image_embeds.size()[:-1],dtype=torch.long,device=image_embeds.device)
            query_tokens = self.query_tokens.expand(image_embeds.shape[0],-1,-1)
            query_outputs = self.qformer_features(query_embeds=query_tokens,
                                                  encoder_hidden_states=image_embeds,
                                                  encoder_attention_mask=image_attention_mask,
                                                  return_dict=True)
            query_output = query_outputs.last_hidden_state
            language_model_inputs = self.language_projection(query_output)
            language_attention_mask = torch.ones(language_model_input.size()[:-1],dtype=torch.long,device=language_model_inputs.device)
            if prompt is None:
                prompt = (torch.LongTensor([[self.config.text_config.bos_token_id]]).repeat(bs,1).to(image_embeds.device))
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt)
            attention_mask = torch.cat([language_attention_mask,attention_mask.to(language_attention_mask.device)],dim=1)
            input_embeds = self.get_text_features(prompt)
            #concatenate query_embeddinf with prompt embedding
            input_embeds = torch.cat([language_model_inputs,input_embeds.to(language_model_inputs.device)],dim=1)
            outputs = self.langauge_model.generate(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                **generate_kwargs)
        input_embeds = self.get_text_features(prompt)
        outputs = self.langauge_model.generate(
                input_embeds=input_embeds,
                attention_mask=attention_mask,
                **generate_kwargs)
        return outputs
        
        
    
               

