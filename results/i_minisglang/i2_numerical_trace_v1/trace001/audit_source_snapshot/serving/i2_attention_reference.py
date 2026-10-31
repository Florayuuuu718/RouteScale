"""Independent contiguous, bottom-right causal math for captured Q/K/V."""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def causal_math(q, k, v, query_start, dtype=torch.float32):
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError('Q/K/V must be token, head, dim tensors')
    if q.shape[2] != k.shape[2] or q.shape[1] % k.shape[1] or query_start < 0 or query_start + len(q) > len(k):
        raise ValueError('Invalid GQA or logical query position')
    mask=torch.arange(len(k),device=q.device)[None,:] <= torch.arange(query_start,query_start+len(q),device=q.device)[:,None]
    with sdpa_kernel(SDPBackend.MATH):
        result=F.scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0).to(dtype),
            k.transpose(0,1).unsqueeze(0).to(dtype),v.transpose(0,1).unsqueeze(0).to(dtype),
            attn_mask=mask,dropout_p=0,enable_gqa=True)
    return result.squeeze(0).transpose(0,1).contiguous()


def tensor_metrics(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError('Aligned logical tensors must have equal shapes')
    a,b=actual.float(),expected.float()
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(finite=False,exact=False,max_abs=None,relative_l2=None,different_elements=None)
    delta=a-b
    return dict(finite=True,exact=torch.equal(actual,expected),max_abs=float(delta.abs().max()),
        relative_l2=float(delta.norm()/b.norm().clamp_min(1e-12)),different_elements=int((actual!=expected).sum()))


def metadata_checks(meta, input_ids, cached, num_pages):
    indices=meta['indices'].long()
    return dict(cache_hit=meta['cached']==cached,extend=meta['extend']==len(input_ids)-cached,
        query_lengths=meta['q_indptr']==[0,len(input_ids)-cached],
        kv_lengths=meta['kv_indptr']==[0,len(input_ids)] and meta['seq_lens']==[len(input_ids)],
        positions=torch.equal(meta['positions'].long(),torch.arange(cached,len(input_ids))),
        input_tokens=torch.equal(meta['input_ids'].long(),torch.tensor(input_ids[cached:])),
        indices_unique=len(indices)==len(input_ids) and len(set(indices.tolist()))==len(indices) and bool(((indices>=0)&(indices<num_pages)).all()),
        write_locations=torch.equal(meta['out_loc'].long(),indices[cached:]))
