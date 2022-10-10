import torch
from einops import rearrange 
from torch.nn import functional as F

""" Slightly adapted from  https://github.com/karpathy/minGPT/blob/master/mingpt/model.py """

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

# @torch.no_grad()
# def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    # """
    # take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    # the sequence, feeding the predictions back into the model each time. Clearly the sampling
    # has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    # of block_size, unlike an RNN that has an infinite context window.
    # """
    # block_size = model.get_block_size()
    # model.evaluate()
    # for k in range(steps):
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        # logjts, _ = model(x_cond)
        # # pluck the logits at the final step and scale by temperature
        # logits = logits[:, -1, :] / temperature
        # # optionally crop probabilities to only the top k options
        # if top_k is not None:
            # logits = top_k_logits(logits, top_k)
        # # apply softmax to convert to probabilities
        # probs = nn.functional.softmax(logits, dim=-1)
        # # sample from the distribution or take the most likely
        # if sample:
            # ix = torch.multinomial(probs, num_samples=1)
        # else:
            # _, ix = torch.topk(probs, k=1, dim=-1)
        # # append to the sequence and continue
        # x = torch.cat((x, ix), dim=1)

    # return x

#temperature=1.0, sample=False, top_k=None,
def sample_from_logits(logits, temperature=1.0, top_k=None, sample=False):
    """ Samples from logits with top_k and temperature.
    Input is of shape [batch_size, time, nb_books, softmax_size]"""

    batch = logits.shape[0]
    logits = logits[:, -1, ...] # Take last time step.
    # Get logits at the final step, put book dimension in batch size
    logits = rearrange(logits, 'batch books emb -> (batch books) emb')
    # scale by temperature
    logits = logits / temperature
    # optionally crop probabilities to only the top k options
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    # apply softmax to convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution or take the most likely
    if sample:
        iz = torch.multinomial(probs, num_samples=1)
    else:
        _, iz = torch.topk(probs, k=1, dim=-1)
    iz = rearrange(iz, '(batch books) one_dim  -> batch one_dim books', batch=batch, one_dim=1)
    return iz
