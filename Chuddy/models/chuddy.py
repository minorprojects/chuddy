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
from Chuddy.utils.util import *
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
            
    

    def prompt_wraps(self,input_ids,img_embeds,target_ids,attention_mask):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        batch_size = input_ids.shape[0]
        bos = torch.ones([batch_size,1],dtype=input_ids.dtype,device=input_ids.device) * self.lm_tokenizer.bos_token_id
        if self.args['freeze_lm']:
            p_after_embeds = self.language_model.model.embed_tokens(input_ids).expand(batch_size,-1,-1)
            bos_embeds = self.language_model.model.embed_tokens(bos)
        else:
            p_after_embeds = self.language_model.model.model.embed_tokens(input_ids).expand(batch_size,-1,-1)
            bos_embeds = self.language_model.model.model.embed_tokens(bos)
        if img_embeds is not None:
            p_before = "### Human: <Img>"
            p_before_tokens = self.lm_tokenizer(p_before,return_tensors='pt',add_special_tokens=False).to(self.device)
            if self.args['freeze_lm']:
                p_before_embeds = self.language_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size,-1,-1)
            else:
                p_before_embeds = self.language_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size,-1,-1)
            input_embeds = torch.cat([p_after_embeds,p_before_embeds,img_embeds,bos_embeds],dim=1).to(self.device)
            empty_targets = (
                torch.ones([batch_size,1 + p_before_embeds.size()[1]+1],dtype=torch.long).to(self.device).fill_(-100)
                )
            targets = torch.cat([empty_targets,target_ids],dim=1).to(self.device)
            assert input_embeds.size()[1] == targets.size()[1]
            atts_prefix = torch.ones([batch_size,1+p_before_embeds.size()[1]+1],dtype=torch.long).to(self.device)
            attention_mask = torch.cat([atts_prefix,attention_mask],dim=1).to(self.device)
            assert attention_mask.size() == targets.size()
        else:
            p_before = "### Human"
            p_before_tokens = self.lm_tokenizer(p_before,return_tensors="pt",add_special_token=False).to(self.device)
            if self.args['freeze_lm']:
                p_before_embeds = self.language_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size,-1,-1)
            else:
                p_before_embeds = self.language_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size,-1,-1)
            input_embeds = torch.cat([bos_embeds,p_before_embeds,p_after_embeds],dim=1).to(self.device)
            empty_targets = (
                torch.ones([batch_size,1+p_before_embeds.size()[1]],dtype=torch.long).to(self.device).fill_(-100)
                )
            targets = torch.cat([empty_targets,target_ids],dim=1).to(self.device)
            assert input_embeds.size()[1] == targets.size()[1]
            atts_prefix = torch.ones([batch_size,1+p_before_embeds.size()[1]],dtype=torch.long).to(self.device)
            attention_mask = torch.cat([atts_prefix,attention_mask],dim=1).to(self.device)
            assert attention_mask.size() == target.size()
        return input_embeds,targets,attention_mask

    def _train_with_mode(self,
                         texts,
                         img_embeds=None,
                         modality='text',
                         num_gen_tokens='8',
                         text_hidden_fcs=None,
                         gen_token_idx=None,
                         text_emb_layers=None,
                         text_prompt_embeddings=None,
                         loss_scale=1.0,
                         stage=2):
        if stage == 2:
            input_ids,target_ids,attention_mask = process_batch_stage_2(
                self.lm_tokenizer,
                texts,
                self.max_len,
                num_gen_tokens,
                modality
                )
        elif stage == 3:
            input_ids,target_ids,attention_mask = process_batch_stage_3(
                self.lm_tokenizer,
                texts,
                self.max_len,
                self.args['num_gen_img_tokens'],
                self.args['num_gen_video_tokens'],
                self.args['num_gen_audio_tokens']
                )
        else:
            raise NotImplementedError 
        input_embeds,targets,attention_mask = self.prompt_wrap(img_embeds,
                                                               input_ids,
                                                               target_ids,
                                                               attention_mask)
        outputs = self.language_model(inputs_embeds=input_embeds,
                                      attention_mask=attention_mask,
                                      return_dict=True,
                                      output_hidden_states=True,
                                      labels=targets,)
        loss = output.loss
        
        #calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits,dim=-1)[:,1:-1]
        labels = targets[:,2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item()+1.0)
        
        if modality == 'text':
            return loss,gen_acc,torch.zeros_like(loss)
        else:
            hidden_state = []
            #i don't know why this code works, but i kow it does
            start_pos = (targets == gen_token_idx[0]).nonzero(as_tuple=False)[:,1].tolist()
            end_pos = (targets == gen_token_idx[-1]).nonzero(as_tuple=False)[:,1].tolist()
            assert 0 < len(start_pos) == len(end_pos) == input_ids.size(0) and len(end_pos) > 0,(start_pos,end_pos)
            for idx, fc_layer in zip(text_emb_layers,text_hidden_fcs):
                hidden_embedding = []
                input_embedding = []
                for b,(s,e) in enumerate(zip(start_pos,end_pos)):
                    assert e-s+1 == num_gen_tokens, (s,e)
                    hidden_embedding.append(outputs.hidden_states[idx][b,s:e+1,:])
                    input_embedding.append(self.input_embeddings(targets[b,s:e+1]))
                hidden_embedding = torch.stack(hidden_embedding,dim=0)
                input_embedding = torch.stack(input_embedding,dim=0)
                hidden_states.append(fc_layer(hidden_embedding,input_embedding))
            embeddings = torch.stack(hidden_states,dim=-1).sum(dim=-1)
            input_text = [conversation for conversation in texts]
            if modality == 'image':
                mse_loss = l2_loss(embeddings,torch.stack(text_prompt_embeddings,dim=0).to(self.device))
            elif modality == 'video':
                mse_loss = l2_loss(embeddings,torch.stack(text_prompt_embeddings,dim=0))
            else:
                text_prompt_embeddings = torch.stack(text_prompt_embeddings,dim=0).to(self.device)
                assert len(text_prompt_embeddings.shape) == 2, text_prompt_embeddings.shape
                text_prompt_embeddings = text_prompt_embeddings.view(text_prompt_embeddings.size(0),1,
                                                                     text_prompt_embeddings.size(1))
                mse_loss = l2_loss(embeddings,text_prompt_embeddings)
            mse_loss = mse_loss.mean()
            loss += loss_scale * mse_loss
            return loss,gen_acc,mse_loss
    
    def _enc_align_training_stage_1(self,inputs):
        modality = get_modality(inputs['mm_paths'])
        if modality == 'image':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_image(image_paths)
        elif modality == 'video':
            video_paths = inputs['mm_paths']
            mm_embeds,_ = self.encode_video(video_paths)
        elif modality == 'audio':
            audio_paths = inputs['mm_paths']
            mm_embeds ,_ = self.encode_audio(audio_paths)
        else:
            raise NotImplementedError 
        input_ids, target_ids,attention_mask = process_batch_stage_1(self.lm_tokenizer,
                                                                     inputs['output_texts'],
                                                                     self.max_len,
                                                                     self.args['prompt'])
        input_embeds,targets,attention_mask = self.prompt_wrap(mm_embeds,
                                                               input_ids,
                                                               target_ids,
                                                               attention_mask)
        outputs = self.language_model(inputs_embeds=input_embeds,
                                      attention_mask=attention_mask,
                                      return_dict=True,
                                      output_hidden_states=True,
                                      labels=targets,)
        loss = outputs.loss 
        chosen_tokens = torch.max(outputs.logits,dim=-1)[1][:,1:-1]
        labels = targets[:,2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss,gen_acc
    
    def _dec_align_training_stage_2(self,inputs):
        modality = get_modality(inputs["mm_paths"])
        if modality  == "image":
            loss,gen_acc,mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                         modality=modality,
                                                         num_gen_tokens=self.args['num_gen_img_tokens'],
                                                         text_hidden_fcs=self.gen_text_hidden_fcs,
                                                         gen_token_idx=self.args['gen_img_token_idx'],
                                                         text_emb_layers=self.args['text_emb_to_img_layers'],
                                                         text_prompt_embeddings=inputs['caption_embs'],
                                                         stage=self.stage)
        elif modality  == "video":
             loss,gen_acc,mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                          modality=modality,
                                                          num_gen_tokens=self.args['num_gen_video_tokens'],
                                                          text_hidden_fcs=self.gen_text_hidden_fcs_video,
                                                          gen_token_idx=self.args['gen_video_token_idx'],
                                                          text_emb_layers=self.args['text_emb_to_video_layers'],
                                                          text_prompt_embeddings=inputs['caption_embs'],
                                                          stage=self.stage)
        elif modality  == "audio":
             loss,gen_acc,mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                          modality=modality,
                                                          num_gen_tokens=self.args['num_gen_audio_tokens'],
                                                          text_hidden_fcs=self.gen_text_hidden_fcs_audio,
                                                          gen_token_idx=self.args['gen_audio_token_idx'],
                                                          text_emb_layers=self.args['text_emb_to_audio_layers'],
                                                          text_prompt_embeddings=inputs['caption_embs'],
                                                          stage=self.stage)
        else:
            raise NotImplementedError
        return loss,gen_acc,mse_loss
    
    def _instruction_tuning_stage_3(self,inputs):
        loss =0
        gen_acc =0
        mse_loss = []
        target_modality = self.args['modality']
        for modality in target_modality:
                
            if modality  == "image":
                _loss,_gen_acc,_mse_loss = self._train_with_mode(inputs['image_output_texts'],
                                                             None,
                                                             modality,
                                                             self.args['num_gen_img_tokens'],
                                                             self.gen_text_hidden_fcs,
                                                             self.args['gen_img_token_idx'],
                                                             self.args['text_emb_to_img_layers'],
                                                             inputs['image_clip_embs'],
                                                             stage=self.stage)
            elif modality  == "video":
                 _loss,_gen_acc,_mse_loss = self._train_with_mode(inputs['video_output_texts'],
                                                              None,
                                                              modality,
                                                              self.args['num_gen_video_tokens'],
                                                              self.gen_text_hidden_fcs_video,
                                                              self.args['gen_video_token_idx'],
                                                              self.args['text_emb_to_video_layers'],
                                                              inputs['video_clip_embs'],
                                                              stage=self.stage)
            elif modality  == "audio":
                 _loss,_gen_acc,_mse_loss = self._train_with_mode(inputs['audio_output_texts'],
                                                              None,
                                                              modality,
                                                              self.args['num_gen_audio_tokens'],
                                                              self.gen_text_hidden_fcs_audio,
                                                              self.args['gen_audio_token_idx'],
                                                              self.args['text_emb_to_audio_layers'],
                                                              inputs['audio_clip_embs'],
                                                              stage=self.stage)
            else:
                image_paths = inputs['text_path_list']
                img_embeds,_ = self.encode_image(image_paths)
                _loss1,_gen_acc1,_ = self._train_with_mode(inputs['visual_QA_list'],
                                                         img_embeds,
                                                         modality=modality,
                                                         stage=self.stage)
                _loss2,_gen_acc2,_ = self._train_with_mode(inputs['output_texts'],
                                                           None,
                                                           modality=modality,
                                                           stage=self.stage)
                _loss = _loss1 + _loss2
                _gen_acc = (_gen_acc1 + _gen_acc2) / 2
            loss += _loss 
            gen_acc += _gen_acc 
            mse_loss.append(_mse_loss)
        gen_acc = gen_acc / len(target_modality)
        return loss,gen_acc,mse_loss
    
    def forward(self,inputs):
        loss= 0
        gen_acc= 0
        mse_loss=None
        
        if self.stage == 1:
            loss,gen_acc = self._enc_align_training_stage_1(inputs)
        elif self.stage == 2:
            loss,gen_acc = self._dec_align_training_stage_2(inputs)
        elif self.stage == 3:
            loss,gen_acc = self._instruction_tuning_stage_3(inputs)
        else:
            raise NotImplementedError(f'stage{self.stage} is not implemented, now it only supports [1,2,3]')
        return loss,gen_acc,mse_loss 
    
    def extract_multimodal_feature(self,inputs):
        features = []
        if inputs['image_paths']:
            image_embeds ,_ = self.encode_image(inputs['image_paths'])
            features.append(image_embeds)
        if inputs['video_paths']:
            video_embeds ,_ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)
        if inputs['audio_paths']:
            audio_embeds ,_ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
            
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return feature_embeds
    
    def _prepare_image_embed(self,text,batch_size):
        pattern =r'Image?(.*?)<\/Image'
        matches = re.findall(pattern,text)
        features = []
        p_before_token = self.lm_tokenizer('<Img>',
                                           add_special_tokens=False,
                                           return_tensors='pt').to(self.device)
        
        p_after_token = self.lm_tokenizer('</Img>',
                                           add_special_tokens=False,
                                           return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.language_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size,-1,-1)
            p_after_embeds = self.language_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size,-1,-1)
        else:
            p_before_embeds = self.language_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size,-1,-1)
            p_after_embeds = self.language_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size,-1,-1)
        for m in matches:
            print('image path:',m)
            if m.startswith('temp'):
                m.os.path.join('../',m)
                print('image path:',m)
            _temp_embedding,_ = self.encode_image([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds,feature_embeds,p_after_embeds],dim=1)
        
    
    def _prepare_video_embed(self,text,batch_size):
        pattern =r'Video?(.*?)<\/Video'
        matches = re.findall(pattern,text)
        features = []
        p_before_token = self.lm_tokenizer('<Vid>',
                                           add_special_tokens=False,
                                           return_tensors='pt').to(self.device)
        
        p_after_token = self.lm_tokenizer('</Vid>',
                                           add_special_tokens=False,
                                           return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.language_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size,-1,-1)
            p_after_embeds = self.language_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size,-1,-1)
        else:
            p_before_embeds = self.language_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size,-1,-1)
            p_after_embeds = self.language_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size,-1,-1)
        for m in matches:
            print('video path:',m)
            if m.startswith('temp'):
                m.os.path.join('../',m)
                print('video path:',m)
            _temp_embedding,_ = self.encode_video([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds,feature_embeds,p_after_embeds],dim=1)
    
    
    def _prepare_image_embed(self,text,batch_size):
        pattern =r'Audio?(.*?)<\/Audio'
        matches = re.findall(pattern,text)
        features = []
        p_before_token = self.lm_tokenizer('<Aud>',
                                           add_special_tokens=False,
                                           return_tensors='pt').to(self.device)
        
        p_after_token = self.lm_tokenizer('</Aud>',
                                           add_special_tokens=False,
                                           return_tensors='pt').to(self.device)
        if self.args['freeze_lm']:
            p_before_embeds = self.language_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size,-1,-1)
            p_after_embeds = self.language_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size,-1,-1)
        else:
            p_before_embeds = self.language_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size,-1,-1)
            p_after_embeds = self.language_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size,-1,-1)
        for m in matches:
            print('audio path:',m)
            if m.startswith('temp'):
                m.os.path.join('../',m)
                print('audio path:',m)
            _temp_embedding,_ = self.encode_audio([m])
            features.append(_temp_embedding)
        feature_embeds = torch.cat(features).sum(dim=0).unsqueeze(0)
        return torch.cat([p_before_embeds,feature_embeds,p_after_embeds],dim=1)
        
     def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        text = prompt + '\n### Assistant:'
        print("text prompt: ", text)
        batch_size = 1
        input_embeds = []
        split_text = re.split(r' <|> ', text)
        for st in split_text:
            if st.startswith('Image>'):
                input_embeds.append(self._prepare_image_embed(st, batch_size))
            elif st.startswith('Audio>'):
                input_embeds.append(self._prepare_audio_embed(st, batch_size))
            elif st.startswith('Video>'):
                input_embeds.append(self._prepare_video_embed(st, batch_size))
            else:
                text_tokens = self.lm_tokenizer(st, add_special_tokens=False, return_tensors='pt').to(self.device)
                bos = torch.ones([batch_size, 1],
                                 dtype=text_tokens.input_ids.dtype,
                                 device=text_tokens.input_ids.device) * self.lm_tokenizer.bos_token_id  # bsz x 1
                if self.args['freeze_lm']:
                    text_embeds = self.language_model.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                    bos_embeds = self.language_model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                else:
                    text_embeds = self.language_model.model.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                    bos_embeds = self.language_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                input_embeds.append(bos_embeds)
                input_embeds.append(text_embeds)
        inputs_embeds = torch.cat(input_embeds, dim=1)  # bsz x (1+s2) x embed_dim
        return inputs_embeds

    def generate_tokens_embeddings(self, inputs, input_embeds, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens and output embeddings that employed to generate images/videos/audios
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
            output_embeddings: output embeddings for synthesizing images
            output_logits: the logits of output tokens
            video_output_embedding: output embeddings for synthesizing video
        """
        with torch.no_grad():  # no tracking history
            # init output with image tokens
            out = None
            output_embeddings = []
            video_output_embedding = []
            audio_output_embedding = []
            output_logits = []

            for i in range(inputs['max_tgt_len']):
                output = self.language_model(inputs_embeds=input_embeds, use_cache=False,
                                          # top_p=top_p,
                                          # temperature=temperature,
                                          # do_sample=True,
                                          output_hidden_states=True)

                if 'image' in self.args['modality']:
                    for idx in self.args['text_emb_to_img_layers']:
                        output_embeddings.append(output.hidden_states[idx])
                if 'video' in self.args['modality']:
                    for idx in self.args['text_emb_to_video_layers']:
                        video_output_embedding.append(output.hidden_states[idx])
                if 'audio' in self.args['modality']:
                    for idx in self.args['text_emb_to_audio_layers']:
                        audio_output_embedding.append(output.hidden_states[idx])

                stop_count = 0
                stop_words_ids = [torch.tensor(x).to(self.device) for x in
                                  inputs['stops_id']]  # '###' can be encoded in two different ways.
                for stop in stop_words_ids:
                    if not (out is None) and torch.all((stop == out[0][-len(stop):])).item():
                        stop_count += 1
                if stop_count >= inputs['ENCOUNTERS']:
                    break

                logits = output.logits[:, -1, :]  # (N, vocab_size)
                output_logits.append(logits)

                # Prevent the model from generating the [IMG1..n] tokens.
                logits[:, self.args['gen_img_token_idx'][1:]] = inputs['filter_value']
                if self.args['gen_img_token_idx'] and self.args['gen_img_token_idx'][0] != -1:
                    if i < inputs['min_word_tokens']:
                        # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
                        logits[:, self.args['gen_img_token_idx']] = inputs['filter_value']

                # Prevent the model from generating the [VID1..n] tokens.
                logits[:, self.args['gen_video_token_idx'][1:]] = inputs['filter_value']
                if self.args['gen_video_token_idx'] and self.args['gen_video_token_idx'][0] != -1:
                    if i < inputs['min_word_tokens']:
                        # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
                        logits[:, self.args['gen_video_token_idx']] = inputs['filter_value']

                # Prevent the model from generating the [AUD1..n] tokens.
                logits[:, self.args['gen_audio_token_idx'][1:]] = inputs['filter_value']
                if self.args['gen_audio_token_idx'] and self.args['gen_audio_token_idx'][0] != -1:
                    if i < inputs['min_word_tokens']:
                        # Eliminate probability of generating [IMG] if this is earlier than min_word_tokens.
                        logits[:, self.args['gen_audio_token_idx']] = inputs['filter_value']

                if inputs['temperature'] == 0.0:
                    if inputs['top_p'] != 1.0:
                        raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                else:
                    logits = logits / inputs['temperature']

                    # Apply top-p filtering.
                    if inputs['top_p'] < 1.0:
                        top_p = inputs['top_p']
                        assert inputs['top_p'] > 0, f'top_p should be above 0, got {top_p} instead.'
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  # (N, D)

                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > inputs['top_p']
                        # Shift the indices to the right to keep also the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        # print('sorted_indices shape: ', sorted_indices.shape)
                        for j in range(sorted_indices.shape[0]):
                            indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
                            logits[j, indices_to_remove] = inputs['filter_value']
                    next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
                    # token_weights = torch.nn.functional.sigmoid(logits.exp())  # (N, vocab_size)
                    # print(logits.sum())
                    # print(token_weights[-20:])
                    # next_token = torch.multinomial(logits, 1)  # (N, 1)
                    # print('next token 2: ', next_token)

                # Force generation of the remaining [IMG] tokens if [IMG0] is generated.
                if next_token.shape[0] == 1 and next_token.item() == self.args['gen_img_token_idx'][0]:
                    next_token = torch.tensor(self.args['gen_img_token_idx'])[None, :].long().to(
                        input_embeds.device)  # (1, num_tokens)
                # Force generation of the remaining [VID] tokens if [VID0] is generated.
                elif next_token.shape[0] == 1 and next_token.item() == self.args['gen_video_token_idx'][0]:
                    next_token = torch.tensor(self.args['gen_video_token_idx'])[None, :].long().to(
                        input_embeds.device)  # (1, num_tokens)
                # Force generation of the remaining [AUD] tokens if [AUD0] is generated.
                elif next_token.shape[0] == 1 and next_token.item() == self.args['gen_audio_token_idx'][0]:
                    next_token = torch.tensor(self.args['gen_audio_token_idx'])[None, :].long().to(
                        input_embeds.device)  # (1, num_tokens)
                else:
                    next_token = next_token.long().to(input_embeds.device)

                if out is not None:
                    out = torch.cat([out, next_token], dim=-1)
                else:
                    out = next_token

                next_embedding = self.input_embeddings(next_token)
                input_embeds = torch.cat([input_embeds, next_embedding], dim=1)

        return out, output_embeddings, output_logits, video_output_embedding, audio_output_embedding

    def generate_images(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40):
        """
        To generate the images based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing videos
        all_gen_idx: the index of [VID0] in the generated_ids
        """
        last_ret_idx = 0
        return_outputs = []
        generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path, torch_dtype=torch.float16).to(
            "cuda")
        for gen_idx in all_gen_idx:
            assert generated_ids[0,
                   gen_idx:gen_idx + self.args['num_gen_img_tokens']].cpu().detach().numpy().tolist() == self.args[
                       'gen_img_token_idx'], (
                generated_ids[0, gen_idx:gen_idx + self.args['num_gen_img_tokens']], self.args['gen_img_token_idx'])
            raw_emb = embeddings[:, gen_idx:gen_idx + self.args['num_gen_img_tokens'], :]  # (1, 8, 4096)

            # Produce generation embedding.
            gen_prefix = ' '.join([f'[IMG{i}]' for i in range(self.args['num_gen_img_tokens'])])
            gen_prefx_ids = self.lm_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="pt").input_ids.to(self.device)
            gen_prefix_embs = self.input_embeddings(gen_prefx_ids)  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_fcs[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)

            if gen_emb.shape[1] != 77:
                bs = gen_emb.shape[0]
                clip_emb = 768
                gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T_I_V_A.txt, 768)
                seq_len = gen_emb.shape[1]
                gen_emb = torch.cat([gen_emb, torch.zeros((bs, 77 - seq_len, clip_emb), device=gen_emb.device,
                                                          dtype=gen_emb.dtype)], dim=1)

            image_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps).images

            caption = \
                self.lm_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(image_outputs)
        return return_outputs

    def generate_videos(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40, height=320, width=576, num_frames=16):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing videos
        all_gen_idx: the index of [VID0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = TextToVideoSDPipeline.from_pretrained(self.vd_ckpt_path, torch_dtype=torch.float16).to(
            "cuda")
        for gen_idx in all_gen_idx:
            assert generated_ids[0,
                   gen_idx:gen_idx + self.args['num_gen_video_tokens']].cpu().detach().numpy().tolist() == \
                   self.args[
                       'gen_video_token_idx'], (
                generated_ids[0, gen_idx:gen_idx + self.args['num_gen_video_tokens']],
                self.args['gen_video_token_idx'])
            raw_emb = embeddings[:, gen_idx:gen_idx + self.args['num_gen_video_tokens'], :]  # (1, 8, 4096)
            # print(f'gen_idx: {gen_idx}')
            # print('4', raw_emb.size())
            # assert len(self.args['text_emb_to_video_layers']) == 1

            # Produce generation embedding.
            gen_prefix = ' '.join([f'[VID{i}]' for i in range(self.args['num_gen_video_tokens'])])
            gen_prefx_ids = self.lm_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="pt").input_ids.to(self.device)
            gen_prefix_embs = self.input_embeddings(gen_prefx_ids)  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_fcs_video[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)

            if gen_emb.shape[1] != 77:
                print(f"Padding {gen_emb.shape} with zeros")
                bs = gen_emb.shape[0]
                clip_emb = 768
                gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T_I_V_A.txt, 768)
                seq_len = gen_emb.shape[1]
                gen_emb = torch.cat([gen_emb, torch.zeros((bs, 77 - seq_len, clip_emb), device=gen_emb.device,
                                                          dtype=gen_emb.dtype)], dim=1)
                print('Padded to', gen_emb.shape)

            video_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps, height=height,
                                             width=width, num_frames=num_frames).frames
            caption = \
                self.lm_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(video_outputs)
        return return_outputs

    def generate_audios(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40, audio_length_in_s=5.0):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing videos
        all_gen_idx: the index of [VID0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = AudioLDMPipeline.from_pretrained(self.ad_ckpt_path, torch_dtype=torch.float16).to("cuda")
        for gen_idx in all_gen_idx:
            assert generated_ids[0,
                   gen_idx:gen_idx + self.args['num_gen_audio_tokens']].cpu().detach().numpy().tolist() == \
                   self.args[
                       'gen_audio_token_idx'], (
                generated_ids[0, gen_idx:gen_idx + self.args['num_gen_audio_tokens']],
                self.args['gen_audio_token_idx'])
            raw_emb = embeddings[:, gen_idx:gen_idx + self.args['num_gen_audio_tokens'], :]  # (1, 8, 4096)
            # print(f'gen_idx: {gen_idx}')
            # print('raw_emb 4', raw_emb.size())
            # assert len(self.args['text_emb_to_video_layers']) == 1

            # Produce generation embedding.
            gen_prefix = ' '.join([f'[AUD{i}]' for i in range(self.args['num_gen_audio_tokens'])])
            gen_prefx_ids = self.lm_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="pt").input_ids.to(self.device)
            gen_prefix_embs = self.input_embeddings(gen_prefx_ids)  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_fcs_audio[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)
            # print('gen_emb size:', gen_emb.size())
            bs = gen_emb.shape[0]
            hid_emb_size = gen_emb.shape[2]
            gen_emb = gen_emb.view(bs, hid_emb_size)

            audio_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             audio_length_in_s=audio_length_in_s).audios[0]
            caption = \
                self.lm_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(audio_outputs)
        return return_outputs

    def generate(self, inputs):
        """
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature, Used to modulate logit distribution.
                'modality_embeds': None or torch.tensor,
                'modality_cache': save the image cache,

                'filter_value': Value to assign to tokens that should never be generated,
                'min_word_tokens': Minimum number of words to generate before allowing a [IMG] output.
                'gen_scale_factor': float = 1.0,
                'stops_id': the default value is [[835], [2277, 29937]] the stop token is '###', which has two types of tokenization ways, [835] and [2277, 29937]
                'ENCOUNTERS': the times that the generated sentence will be ended.

                'load_sd': whether use SD for image generation
                'max_num_imgs': Maximum number of images to return in one generation pass.
                'guidance_scale_for_img': the guidance ratio of conditioner, if it is None, the default value will be applied in SD
                'num_inference_steps_for_img': the number of inference step for image generation in the stable diffusion model

                'load_vd': whether use VD for video generation
                'max_num_vids': Maximum number of videos to return in one generation pass.
                'guidance_scale_for_vid': the guidance ratio of conditioner, if it is None, the default value will be applied in VD
                'num_inference_steps_for_vid': the number of inference step for video generation in the stable diffusion model
                'height': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The height in pixels of the generated video.
                'width': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The width in pixels of the generated video.
                'num_frames': (`int`, *optional*, defaults to 16):
                        The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                        amounts to 2 seconds of video.

                'load_ad': whether use AD for audio generation
                'max_num_auds': Maximum number of audios to return in one generation pass.
                'guidance_scale_for_aud': the guidance ratio of conditioner, if it is None, the default value will be applied in AD
                'num_inference_steps_for_aud': the number of inference step for audio generation in the stable diffusion model
                'audio_length_in_s': the seconds for generated audio length
            }
        """
        # init output with image tokens

        input_embeds = self.prepare_generation_embedding(inputs)
        generated_ids, generated_embeddings, _, generated_video_embeddings, generated_audio_embeddings = self.generate_tokens_embeddings(
            inputs, input_embeds)

        return_outputs = []

        # Find up to max_num_rets [IMG] tokens, and their corresponding scores.
        all_gen_img_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_img_token_idx'][0]) if x][
                          :inputs['max_num_imgs']]
        print('all_gen_img_idx: ', all_gen_img_idx)

        # Find up to max_num_rest [VID] tokens, and their corresponding scores.
        all_gen_vid_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_video_token_idx'][0]) if x][
                          :inputs['max_num_vids']]
        print('all_gen_vid_idx: ', all_gen_vid_idx)

        # Find up to max_num_rest [AUD] tokens, and their corresponding scores.
        all_gen_aud_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_audio_token_idx'][0]) if x][
                          :inputs['max_num_auds']]
        print('all_gen_aud_idx: ', all_gen_aud_idx)

        if len(all_gen_img_idx) == 0 and len(all_gen_vid_idx) == 0 and len(all_gen_aud_idx) == 0:
            # No [IMG], [VID], [AUD] tokens.
            caption = self.lm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # return_outputs.append(truncate_caption(caption))
            return_outputs.append(caption)
        else:
            if len(all_gen_img_idx) > 0:
                embeddings = generated_embeddings[-1][:, input_embeds.shape[1]:]
                # embeddings = embeddings[:, :trunc_idx] if trunc_idx > 0 else embeddings
                # sd_pipe = StableDiffusionPipeline.from_pretrained(self.args['image_generation_ckpt_path'],
                #                                                   torch_dtype=torch.float16).to(self.device)
                img_outputs = self.generate_images(generated_ids, embeddings, all_gen_img_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_img'],
                                                   num_inference_steps=inputs['num_inference_steps_for_img'],
                                                   )
                return_outputs.append({'img': img_outputs})
            if len(all_gen_vid_idx) > 0:
                video_embeddings = generated_video_embeddings[-1][:, input_embeds.shape[1]:]
                # video_embeddings = video_embeddings[:, :trunc_idx] if trunc_idx > 0 else video_embeddings
                # vd_pipe = TextToVideoSDPipeline.from_pretrained(self.args['video_generation_ckpt_path'],
                #                                                 torch_dtype=torch.float16).to(self.device)
                vid_outputs = self.generate_videos(generated_ids, video_embeddings, all_gen_vid_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_vid'],
                                                   num_inference_steps=inputs['num_inference_steps_for_vid'],
                                                   height=inputs['height'], width=inputs['width'],
                                                   num_frames=inputs['num_frames'])
                return_outputs.append({'vid': vid_outputs})

            if len(all_gen_aud_idx) > 0:
                audio_embeddings = generated_audio_embeddings[-1][:, input_embeds.shape[1]:]
                # audio_embeddings = audio_embeddings[:, :trunc_idx] if trunc_idx > 0 else audio_embeddings
                # ad_pipe = AudioLDMPipeline.from_pretrained(self.args['audio_generation_ckpt_path'],
                #                                            torch_dtype=torch.float16).to(self.device)
                aud_outputs = self.generate_audios(generated_ids, audio_embeddings, all_gen_aud_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_aud'],
                                                   num_inference_steps=inputs['num_inference_steps_for_aud'],
                                                   audio_length_in_s=inputs['audio_length_in_s'])
                return_outputs.append({'aud': aud_outputs})

        return return_outputs
    
    
               

