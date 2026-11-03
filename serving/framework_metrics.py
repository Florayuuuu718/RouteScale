"""Read the pinned SGLang exporter; absent metrics remain missing, never zero."""
import math
from collections import Counter
import json
from pathlib import Path
import re

NAMES=dict(running='sglang:num_running_reqs',queued='sglang:num_queue_reqs',
    processed='sglang:num_requests_total',aborted='sglang:num_aborted_requests_total',
    output_tokens='sglang:generation_tokens_total',throughput='sglang:gen_throughput',
    KV_usage='sglang:token_usage',KV_used='sglang:kv_used_tokens',KV_available='sglang:kv_available_tokens',
    model_GB='sglang:weight_memory_usage_gb',KV_GB='sglang:kv_cache_memory_usage_gb',graph_GB='sglang:graph_memory_usage_gb',
    TTFT_sum='sglang:time_to_first_token_seconds_sum',TTFT_count='sglang:time_to_first_token_seconds_count')


def parse(text):
    samples=[]
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        m=re.fullmatch(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?\s+([^ ]+)(?:\s+\d+)?',line)
        if not m:raise ValueError('Malformed Prometheus sample: '+line[:160])
        value=float(m[3])
        if math.isfinite(value):samples.append(dict(name=m[1],labels=m[2] or '',value=value))
    metrics={key:sum(s['value'] for s in samples if s['name']==name) if any(s['name']==name for s in samples) else None for key,name in NAMES.items()}
    metrics['TTFT_mean_s']=metrics['TTFT_sum']/metrics['TTFT_count'] if metrics['TTFT_count'] else None
    return dict(metrics=metrics,missing=[name for key,name in NAMES.items() if metrics[key] is None],samples=samples)


def client_counts(path):
    """An explicitly selected observation ledger, never an exporter success guess."""
    if path is None:return None
    try:rows=json.loads(Path(path).read_text())
    except FileNotFoundError:return None
    if not isinstance(rows,list) or any(r.get('state') not in ('finished','failed','rejected','cancelled') for r in rows):
        raise ValueError('Expected a terminal client observation ledger')
    counts=Counter(r['state'] for r in rows)
    return dict(offered=len(rows),**{k:counts[k] for k in ('finished','failed','rejected','cancelled')})


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RouteScale SGLang</title>
<style>body{font:16px system-ui;max-width:1100px;margin:36px auto;background:#f4f6f9;color:#162237}h1{font-size:25px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{padding:22px;background:white;border-radius:12px}.value{font-size:30px;font-weight:650}small{color:#52617a}pre{white-space:pre-wrap;font-size:12px}</style>
<h1>RouteScale · SGLang 原生指标</h1><p id="state">读取中</p><div class="grid" id="cards"></div>
<p><small>TTFT为框架首token均值；客户端首文本及终态另存账本。缺失指标显示“未导出”，不补零。显存为框架分配项，实际GPU总占用另由nvidia-smi留样。</small></p><pre id="missing"></pre><h2>本次客户端观测</h2><p id="client">尚无观测账本</p><p id="GPU">GPU总占用待采集</p>
<script>const fields=[['running','运行请求'],['queued','队列请求'],['processed','框架已处理请求'],['aborted','框架中止请求'],['TTFT_mean_s','原生TTFT均值 / s'],['throughput','原生输出吞吐 / token/s'],['output_tokens','原生已生成token'],['KV_usage','KV使用比例'],['KV_used','KV使用token'],['KV_available','KV可用token'],['model_GB','模型显存 / GB'],['KV_GB','KV显存池 / GB'],['graph_GB','图显存 / GB']];async function tick(){try{const r=await fetch('/snapshot');const d=await r.json();document.getElementById('state').textContent='服务状态：'+d.state+' · '+new Date().toLocaleTimeString();const values=d.native?.metrics||{};document.getElementById('client').textContent=d.client?`仅本次已落盘账本：${d.client.offered}个offer · ${d.client.finished}完成 · ${d.client.failed}失败 · ${d.client.rejected}拒绝 · ${d.client.cancelled}取消`:'未指定观测账本；不能由processed推算成功数';document.getElementById('GPU').textContent=d.GPU_used_MiB==null?'GPU总占用未采集':`实际GPU总占用 / MiB（nvidia-smi）：${d.GPU_used_MiB}`;document.getElementById('cards').replaceChildren(...fields.map(([k,label])=>{let e=document.createElement('div');e.className='card';let t=document.createElement('div');t.textContent=label;let v=document.createElement('div');v.className='value';v.textContent=values[k]==null?'未导出':Number(values[k]).toFixed(3);e.append(t,v);return e}));document.getElementById('missing').textContent='缺失项：'+JSON.stringify(d.native?.missing||[])}catch(e){document.getElementById('state').textContent='采集尚未就绪'}}tick();setInterval(tick,2000)</script></html>'''
