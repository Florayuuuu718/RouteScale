"""Offline GPT-2 tokenizer/workload preparation in the unchanged training env."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from serving.backends.nanovllm import ROOT,digest,write_result

DEFAULT_CONFIG=ROOT/"config/serving/h3_routescale.json"


def byte_symbols():
    values=list(range(33,127))+list(range(161,173))+list(range(174,256))
    chars=values.copy()
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256+sum(c>=256 for c in chars))
    return dict(zip(values,map(chr,chars)))


def prepared_bpe(encoding):
    # Same byte/rank conversion contract as the pinned transformers 4.57.6
    # TikTokenConverter, without importing/installing tiktoken in serving env.
    symbols=byte_symbols()
    ranks=encoding._mergeable_ranks
    string=lambda token:"".join(symbols[b] for b in token)
    vocab={string(token):rank for token,rank in ranks.items()}
    candidates=[]
    for token,rank in ranks.items():
        splits=[(token[:i],token[i:]) for i in range(1,len(token))
            if token[:i] in ranks and token[i:] in ranks]
        splits.sort(key=lambda pair:(ranks[pair[0]],ranks[pair[1]]))
        candidates.extend((left,right,rank) for left,right in splits)
    candidates.sort(key=lambda row:row[2])
    return dict(vocab=vocab,merges=[[string(a),string(b)] for a,b,_ in candidates],
        pattern=encoding._pat_str,special_tokens=encoding._special_tokens,
        ranks_sha256=__import__("hashlib").sha256(json.dumps(
            [(base64.b64encode(t).decode(),r) for t,r in sorted(ranks.items(),key=lambda x:x[1])]).encode()).hexdigest())


def prepare(path=DEFAULT_CONFIG):
    import tiktoken
    from serving.workloads import G1_PROMPTS
    config=json.loads(path.read_text())
    output=ROOT/config["results_dir"]/"preparation.json"
    directory=ROOT/config["workload_dir"]
    if output.exists() or any(directory.glob("*.json")):
        raise ValueError("Preserve existing prepared tokenizer/workload evidence")
    encoding=tiktoken.get_encoding("gpt2")
    data=prepared_bpe(encoding)
    texts=[*G1_PROMPTS.values(),"Hello, world!"," I'm we're you've can't I'd", "中文测试：你好！",
        "🙂 café naïve\n\n\t  ","1234567890  001", " a\n b\r\n", "\u0000\u007f\u00a0", "", "    "]
    data["tests"]=[dict(text=text,token_ids=encoding.encode(text),decoded=encoding.decode(encoding.encode(text))) for text in texts]
    pool=encoding.encode(G1_PROMPTS["long"]*12)
    cases=[]
    for spec in config["cases"]:
        requests=[]
        for index,length in enumerate(spec["prompts"]):
            start=(index*47+len(cases)*23)%(len(pool)-length)
            tokens=pool[start:start+length]
            requests.append(dict(request_id=f"r{index}",prompt_token_ids=tokens,output_tokens=spec["output_tokens"]))
        cases.append(dict(case_id=spec["case_id"],requests=requests))
    write_result(data,directory/"tokenizer.json")
    write_result(dict(cases=cases),directory/"requests.json")
    report=dict(schema_version=1,stage="H3",status="pass",config_sha256=digest(path),builder_sha256=digest(Path(__file__)),
        tokenizer_sha256=digest(directory/"tokenizer.json"),workload_sha256=digest(directory/"requests.json"),
        tokenizer_package=tiktoken.__version__,ranks_sha256=data["ranks_sha256"],
        limitations="Prepared offline with existing tiktoken; no serving package installation or model re-export.")
    write_result(report,output)
    print("H3 offline tokenizer/workload: pass",flush=True)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    prepare(parser.parse_args().config)
