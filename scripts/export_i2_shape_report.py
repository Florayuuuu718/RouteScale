"""Export all accepted cells, ratios and figure hashes without selecting winners."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serving.i2_workloads import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();d=args.directory;report=json.loads((d/'audit.json').read_text())
    if not report['passed']:raise ValueError('Accepted timing audit required')
    rows=[]
    for row in report['statistics']:
        result={k:row[k] for k in ('variant','case','phase','throughput_median','throughput_min','throughput_max','first_result_ms_median','process_cv')}
        result.update(prefill_queries=row['processes'][0]['prefill_queries'],prefill_linear_padded_rows_per_projection=row['processes'][0]['prefill_linear_padded_rows_per_projection'])
        rows.append(result)
    for name,values in (('statistics.csv',rows),('comparisons.csv',report['comparisons'])):
        with (d/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    temperatures=[];clocks=[];powers=[];pstates=set()
    for path in d.glob('benchmark_*.json'):
        for row in json.loads(path.read_text())['replay']['cases']:
            fields=[s.strip() for s in row['telemetry_after'].split(',')]
            temperatures.append(float(fields[2]));clocks.append(float(fields[3].split()[0]));powers.append(float(fields[5].split()[0]));pstates.add(fields[6])
    record=dict(scope=report['scope'],source_sha256=digest(Path(__file__)),audit_sha256=digest(d/'audit.json'),
        statistics_cells=len(rows),comparisons=len(report['comparisons']),
        process_cv_min=min(r['process_cv'] for r in rows),process_cv_max=max(r['process_cv'] for r in rows),
        telemetry_after_wave=dict(samples=len(temperatures),temperature_c=[min(temperatures),max(temperatures)],sm_clock_mhz=[min(clocks),max(clocks)],power_w=[min(powers),max(powers)],pstates=sorted(pstates),boundary='after wave/resource audit; does not measure clocks during kernels'),
        artifact_sha256={p.name:digest(p) for p in d.iterdir() if p.suffix in ('.svg','.png','.csv') or p.name=='figures.json'})
    (d/'export.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
