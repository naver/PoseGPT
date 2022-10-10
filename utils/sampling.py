# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import os
import torch
import tqdm
from utils.ae_utils import red
import pickle as pkl
from utils.variable_length import valid_concat_rot_trans
from utils.variable_length import repeat_last_valid
from functools import partial



from evaluate import extract_from_classifier

@torch.no_grad()
def compute_fid(model, data, preparator, device, classif_ckpt,
        temperature=None, top_k=None, cond_steps=0,
        writer=None, current_iter=None, debug=False):
    """
    Sample a full dataset (using data to get the class conditioning), extract features from classifier.
    """
    sample_options = {'class_conditional': model.class_conditional,
                      'seqlen_conditional': model.seqlen_conditional,
                      'temperature': temperature, # Softmax temperature 
                      'top_k': top_k,
                      'cond_steps': cond_steps}
    
    is_training = False
    if (model.training):
        is_training = True
        model.eval()
    real_fid_path = os.path.join('/'.join(classif_ckpt.split('/')[:-3]), 'real_fid_stats.pkl')

    fid_one_batch = partial(fid_sample_one_batch, pose_gpt=model)
    # Forward inputs through classifier and get fid stats 
    output = extract_from_classifier(batch_sampler=fid_one_batch,
                                     data_loader=data,
                                     preparator=preparator,
                                     device=device,
                                     action_model_ckpt=classif_ckpt,
                                     debug=debug,
                                     sample_options=sample_options, real_fid_path=real_fid_path,
                                     summarize=True)
    #model.fid_conclude(is_training)
    if is_training:
        model.train()
    if isinstance(output, dict) and writer is not None:
        assert current_iter
        prefix = "classif"
        for k, v in output.items():
            writer.add_scalar(f"{prefix}/{k}/", v, current_iter)
    else:
        writer.add_scalar('fid/', output, current_iter)

def fid_sample_one_batch(pose_gpt, *, x, valid, actions, device, cond_steps,
        class_conditional, seqlen_conditional, zidx_or_bs, temperature,
        top_k, data_loader, i, preparator):
    """ Sample one batch from the model to evaluate fid on it.
    Used in the inner loop of new_compute_fid"""
    _x, _valid = x, valid
    x, valid, actions = x.to(device), valid.to(device), actions.to(device)
    x, *_ = preparator(x)

    zidx = zidx_or_bs
    if zidx is None or cond_steps > 0 or i == len(data_loader) - 1:
        _, zidx = pose_gpt.vqvae.forward_latents(x, valid, return_indices=True)

    seqlens = valid.sum(1)
    (rotvec, trans), valid, idx = pose_gpt.sample_poses(zidx, actions=actions, seqlens=seqlens, x=x,
            valid=x, temperature=temperature, top_k=top_k, cond_steps=cond_steps, return_index_sample=True)
    sample_valid = _valid.to(device) if pose_gpt.sample_eos_force else valid
    rotvec, trans = [repeat_last_valid(x, sample_valid) for x in [rotvec, trans]]
    poses = torch.cat((rotvec.flatten(2), trans.flatten(2)), dim=-1)
    return poses, sample_valid, zidx


