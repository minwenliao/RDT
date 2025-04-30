import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim, img_token_dim, state_token_dim, 
                 max_lang_cond_len, img_cond_len, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16):
        super(RDTRunner, self).__init__()
        self.dtype=dtype
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.effort_type = config['effort_type']
        # When effort_type is 'fut' or 'his_c_fut', the action output will include effort prediction
        self.action_out_dim = action_dim
        if self.effort_type in ('fut', 'his_c_fut'):
            self.action_out_dim += 14  # Add effort dimension
        
        self.model = RDT(
            output_dim=self.action_out_dim,  # Use expanded output dimension for fut mode
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
            eff_tok_len=1 if self.effort_type in ("his_c", "his_c_fut") else 0
        )

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(
            config['lang_adaptor'], 
            in_features=lang_token_dim, 
            out_features=hidden_size,
            dtype=self.dtype
        )
        self.img_adaptor = self.build_condition_adapter(
            config['img_adaptor'], 
            in_features=img_token_dim, 
            out_features=hidden_size,
            dtype=self.dtype
        
        )
        # A `state` refers to an action or a proprioception vector
        # Calculate input dimensions for state adaptor
        self.state_token_dim = state_token_dim
        self.state_adaptor_in_features = state_token_dim * 2  # Original: state + state mask (indicator)
        if self.effort_type in ("fut", "his_c_fut"):
            self.state_adaptor_in_features = (state_token_dim + config['effort_dim']) * 2
        
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'], 
            in_features=self.state_adaptor_in_features,
            out_features=hidden_size,
            dtype=self.dtype
        )

        self.effort_adaptor = self.build_condition_adapter(
            "mlp2x_gelu", 
            in_features=14 * (10 if self.effort_type in ("his_c", "his_c_fut") else 1), 
            out_features=hidden_size,
            dtype=self.dtype
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim  # Original action dimension without effort
        # When effort_type is 'fut' or 'his_c_fut', the action output will include effort prediction during training
        self.action_out_dim = action_dim
        if self.effort_type in ("fut", "his_c_fut"):
            self.action_out_dim += 14  # Add effort dimension

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()] + 
            [p.numel() for p in self.lang_adaptor.parameters()] + 
            [p.numel() for p in self.img_adaptor.parameters()] + 
            [p.numel() for p in self.state_adaptor.parameters()]))
    
    def build_condition_adapter(
        self, projector_type, in_features, out_features,dtype):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features,dtype=dtype)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features,dtype=dtype)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features,dtype=dtype))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens, efforts=None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)
        adpated_effort = None
        if efforts is not None:
            adpated_effort = self.effort_adaptor(efforts)
            # adpated_img = torch.cat([adpated_img, adpated_effort], dim=1)

        return adpated_lang, adpated_img, adpated_state, adpated_effort

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, 
                           state_traj, action_mask, ctrl_freqs, effort_token):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(
            size=(state_traj.shape[0], self.pred_horizon, self.action_out_dim), 
            dtype=dtype, device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)
    
        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)
        
        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            if effort_token is not None:
                state_action_traj = torch.cat([state_traj, effort_token, action_traj], dim=1)
            else:
                state_action_traj = torch.cat([state_traj, action_traj], dim=1)
            
            # Predict the model output
            model_output = self.model(state_action_traj, ctrl_freqs,
                                    t.unsqueeze(-1).to(device),
                                    lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(
                model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, 
                     state_tokens, action_gt, action_mask, ctrl_freqs,
                     efforts=None,
                    ) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device  

        if self.effort_type in ("fut", "his_c_fut"):
            # padding
            state_tokens = torch.cat([
                state_tokens,
                torch.zeros((batch_size, 1, 14), device=device, dtype=state_tokens.dtype)
            ], dim=-1)
            action_mask = torch.cat([
                action_mask,
                torch.zeros((batch_size, 1, 14), device=device, dtype=action_mask.dtype)
            ], dim=-1)

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            action_gt.shape, dtype=action_gt.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            action_gt, noise, timesteps)
        
        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj, effort_token = self.adapt_conditions(
            lang_tokens, img_tokens, state_action_traj, efforts)
        if effort_token is not None:
            state_action_traj = torch.cat([state_action_traj[:,:1], effort_token, state_action_traj[:,1:]], dim=1)
        # Predict the denoised result
        pred = self.model(state_action_traj, ctrl_freqs, 
                          timesteps, lang_cond, img_cond, 
                          lang_mask=lang_attn_mask)

        pred_type = self.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target)
        return loss
    
    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens,
                       action_mask, ctrl_freqs, efforts=None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        # if self.effort_type in ("fut", "his_c_fut"):
        #     state_tokens = torch.cat([state_tokens, torch.zeros((state_tokens.shape[0], 1, 14), dtype=state_tokens.dtype, device=state_tokens.device)], dim=-1)
        #     action_mask = torch.cat([action_mask, torch.zeros((action_mask.shape[0], 1, 14), dtype=action_mask.dtype, device=action_mask.device)], dim=-1)

        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj, effort_token = self.adapt_conditions(
            lang_tokens, img_tokens, state_tokens, efforts)
        
        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond, lang_attn_mask, img_cond, 
            state_traj, action_mask, ctrl_freqs, effort_token
        )
        
        # For fut mode, only return the action part, not the effort predictions
        # if self.effort_type in ("fut", "his_c_fut"):
        #     action_pred = action_pred[..., :self.action_dim]
        
        return action_pred
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """load pretrain weights in different shape models. COMMENT THIS DURING INFERENCE"""
        if self.effort_type not in ("his_c", "fut", "his_c_fut"):
            # Normal loading for non-fut modes
            return super().load_state_dict(state_dict, strict)
            
        # For fut mode, we need special handling of the final layer and state adaptor
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == "model.final_layer.ffn_final.fc2.weight" and "fut" in self.effort_type:
                # Handle final layer weight
                new_weight = torch.zeros(self.action_out_dim, value.size(1), dtype=value.dtype, device=value.device)
                new_weight[:self.action_dim] = value
                new_weight[self.action_dim:].normal_(mean=0.0, std=0.02)
                new_state_dict[key] = new_weight
            elif key == "model.final_layer.ffn_final.fc2.bias" and "fut" in self.effort_type:
                # Handle final layer bias
                new_bias = torch.zeros(self.action_out_dim, dtype=value.dtype, device=value.device)
                new_bias[:self.action_dim] = value
                new_bias[self.action_dim:].zero_()
                new_state_dict[key] = new_bias
            elif key == "state_adaptor.0.weight" and "fut" in self.effort_type:
                # NOTE bug here
                # Handle state adaptor input layer weight (works for both Linear and first layer of MLP)
                new_weight = torch.zeros(value.size(0), self.state_adaptor_in_features, 
                                       dtype=value.dtype, device=value.device)
                new_weight.normal_(mean=0.0, std=0.02)
                # Copy original weights for state dimensions
                new_weight[:, :self.action_dim] = value[:, :self.action_dim]
                new_weight[:, self.action_dim + 14:self.action_dim + 14 + self.action_dim] = value[:, self.action_dim:]
                new_state_dict[key] = new_weight
            elif key == "model.x_pos_embed" and "his_c" in self.effort_type:
                # Handle positional embedding when eff_tok_len changes from 0 to 1
                # Original shape is [1, horizon+3, hidden_size], new shape should be [1, horizon+3+1, hidden_size]
                old_size = value.size(1)
                new_size = old_size + 1  # Adding 1 for eff_tok_len
                new_embed = torch.zeros(1, new_size, value.size(2), dtype=value.dtype, device=value.device)
                new_embed.normal_(mean=0.0, std=0.02)
                # Copy the original embeddings
                new_embed[:, :-self.pred_horizon-1] = value[:, :-self.pred_horizon]
                new_embed[:, -self.pred_horizon:] = value[:, -self.pred_horizon:]
                new_state_dict[key] = new_embed
            else:
                new_state_dict[key] = value
        
        # Load the modified state dict
        return super().load_state_dict(new_state_dict, strict)
