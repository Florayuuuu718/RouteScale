import fcntl,json,subprocess,time,traceback
from pathlib import Path
root=Path.cwd();cli='/mnt/d/APP/Docker/resources/bin/docker.exe';prior=root/'runs/tmp/framework_container_result.json';start=time.monotonic()
while not prior.exists():
 if time.monotonic()-start>6000:raise TimeoutError('Container healthy prerequisite')
 time.sleep(5)
healthy=json.loads(prior.read_text());assert healthy['passed'];image=healthy['image_id'];commit=healthy['source_commit'];name='routescale-m5-stop-'+commit[:7];cache='routescale-m5-cache-'+commit[:7]
model=r'type=bind,source=D:\RouteScale-local-delivery-20261103\models\Qwen3-0.6B,target=/models/Qwen3-0.6B,readonly'
cmd=[cli,'run','--detach','--name',name,'--gpus','device=0','--shm-size=1g','--env','ROUTESCALE_RUN_ID=stop001','--env','ROUTESCALE_SOURCE_COMMIT='+commit,'--mount',model,'--volume',cache+':/root/.cache','--publish','127.0.0.1:21971:1971','--publish','127.0.0.1:21972:1972',image]
result=dict(passed=False,command=cmd,image_id=image,source_commit=commit,name=name)
def inside(code):return subprocess.check_output([cli,'exec',name,'python3','-c',code],text=True,stderr=subprocess.STDOUT,timeout=20)
with (root/'runs/tmp/h2_runner.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()=='0 MiB'
 cid=None;client=None
 try:
  cid=subprocess.check_output(cmd,text=True,timeout=30).strip();result['container_id']=cid
  deadline=time.monotonic()+615
  while time.monotonic()<deadline:
   state=json.loads(inside("import json,pathlib;p=pathlib.Path('/state/stop001/status.json');d=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({k:d.get(k) for k in ['state','supervisor_pid']}))"))
   if state.get('state')=='ready':break
   if state.get('state')=='failed':raise RuntimeError('Default entrypoint failed before readiness')
   time.sleep(1)
  else:raise TimeoutError('Default entrypoint readiness')
  assert state['supervisor_pid']==1;result['actual_supervisor_PID1']=True
  code="""import asyncio,json,pathlib,time
from serving.framework_client import CompletionClient
from serving.online_client import OnlineRequest
w=json.loads(pathlib.Path('results/i_minisglang/i3_online_v2/workload002/workload.json').read_text());payload=dict(w['requests'][0]['payload'],model='RouteScale-Qwen3-0.6B',max_tokens=128)
def sink(e):
 if e['event']=='first_text':pathlib.Path('/state/stop_client_started.json').write_text(json.dumps(e))
rows=asyncio.run(CompletionClient('http://127.0.0.1:1971',concurrency=1,sink=sink).run([OnlineRequest('docker_stop_inflight',payload,timeout_s=30)]));pathlib.Path('/state/stop_client.json').write_text(json.dumps(rows))
"""
  log=(root/'runs/tmp/framework_docker_stop_client.log').open('wb');client=subprocess.Popen([cli,'exec',name,'python3','-c',code],stdout=log,stderr=subprocess.STDOUT)
  deadline=time.monotonic()+15
  while time.monotonic()<deadline:
   flag=inside("import pathlib;print(pathlib.Path('/state/stop_client_started.json').exists())").strip()
   if flag=='True':break
   if client.poll() is not None:raise RuntimeError('Client ended before active stop trigger')
   time.sleep(.1)
  else:raise TimeoutError('No real first text before docker stop')
  result['native_loads_at_stop']=json.loads(inside("import json,urllib.request;print(urllib.request.urlopen('http://127.0.0.1:1971/v1/loads').read().decode())"))
  assert result['native_loads_at_stop']['loads'][0]['num_running_reqs']>=1
  before=time.monotonic();stop=subprocess.run([cli,'stop','--time','30',cid],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=45);result.update(docker_stop_returncode=stop.returncode,docker_stop_s=time.monotonic()-before)
  assert stop.returncode==0;client.wait(timeout=15);log.close();result['client_returncode']=client.returncode
  inspect=json.loads(subprocess.check_output([cli,'inspect',cid],text=True,timeout=20))[0]
  result.update(container_exit_code=inspect['State']['ExitCode'],readonly_model=not next(m for m in inspect['Mounts'] if m['Destination']=='/models/Qwen3-0.6B')['RW'])
  dest=Path('/mnt/d/RouteScale-local-delivery-20261103/container_stop001');dest.mkdir(exist_ok=False)
  subprocess.run([cli,'cp',cid+':/state',r'D:\RouteScale-local-delivery-20261103\container_stop001'],check=True,timeout=60)
  state=json.loads((dest/'state/stop001/status.json').read_text());rows=json.loads((dest/'state/stop_client.json').read_text());result['native_status']=state;result['client_rows']=rows
  assert state['state']=='stopped' and not state['forced_cleanup'] and state['GPU_end']=='0 MiB' and rows[0]['state']=='finished' and inspect['State']['ExitCode']==0 and result['readonly_model']
  result['passed']=True
 except BaseException:result['error']=traceback.format_exc();print(result['error'],flush=True)
 finally:
  if cid and not result['passed']:subprocess.run([cli,'stop','--time','30',cid],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=45)
  result['GPU_end']=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],text=True).strip()
  (root/'runs/tmp/framework_docker_stop_result.json').write_text(json.dumps(result,indent=2)+'\n')
print('Actual default-entrypoint docker stop',result['passed'],flush=True)
if not result['passed']:raise SystemExit(1)
