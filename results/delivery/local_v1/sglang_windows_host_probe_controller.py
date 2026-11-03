import json,subprocess,time
from pathlib import Path
code="""$ErrorActionPreference='Stop'; try {$r=Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:21971/ready -TimeoutSec 2; $m=Invoke-RestMethod -Uri http://127.0.0.1:21971/v1/models -TimeoutSec 3; $d=Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:21972/ -TimeoutSec 3; [ordered]@{ready_status=$r.StatusCode; model_ids=@($m.data | ForEach-Object {$_.id}); dashboard_status=$d.StatusCode; dashboard_has_cards=$d.Content.Contains('cards'); host='Windows127.0.0.1';ports=@(21971,21972)} | ConvertTo-Json -Compress} catch {exit 3}"""
end=time.monotonic()+650
while time.monotonic()<end:
 r=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',code],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
 if r.returncode==0:
  d=json.loads(r.stdout);d['passed']=d['ready_status']==d['dashboard_status']==200 and d['dashboard_has_cards'] and 'RouteScale-Qwen3-0.6B' in d['model_ids'];Path('runs/tmp/framework_windows_host_probe.json').write_text(json.dumps(d,indent=2)+'\n');print(d,flush=True);raise SystemExit(0 if d['passed'] else 1)
 time.sleep(4)
raise TimeoutError('Published Windows localhost endpoints never became ready')
