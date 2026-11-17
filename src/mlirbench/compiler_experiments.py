"""Portable replay and fresh compiler experiments; historical records stay immutable."""
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

BACKEND_VERSION = '3.12.0rc20260916'
PASSES = {'simplification_pipeline':['--torch-simplification-pipeline'],
          'reduce_op_variants':['--torch-reduce-op-variants'],
          'simplify_shapes':['--torch-simplify-shape-calculations'],
          'decompose_complex':['--torch-decompose-complex-ops']}


def sha(raw): return hashlib.sha256(raw).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))


def truth(value):
    if value in (True,'True'): return True
    if value in (False,'False'): return False
    raise ValueError('Missing or invalid boolean observation: '+repr(value))


def source_lookup(root):
    result={}
    with (Path(root)/'data/processed/MLIRBench/master/master.jsonl').open() as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line)
            for field in ('input_mlir','mutated_mlir'):
                text=row[field].strip(); result[sha(text.encode())]=text
    return result


def scan_rows(root,dialect):
    name=('DT1_ELEMENTWISE/outputs/fusion_scan_results.csv' if dialect=='linalg'
          else 'DT1_TORCH_OPT/outputs/torch_opt_scan.csv')
    path=Path(root)/'data/downstream/records'/name
    rows=read_csv(path)
    if len({r['dt1_id'] for r in rows})!=len(rows): raise ValueError('Duplicate compiler scan ID')
    return rows,path


def instructions(text):
    """Same instruction/pseudo-instruction definition as the archived RISC-V driver."""
    found=[]
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith(('.','#','//',';')) or line.endswith(':'): continue
        line=re.split(r'\s+#|\s+//',line,maxsplit=1)[0].strip()
        if re.match(r'^[A-Za-z][A-Za-z0-9_.]*(?:\s|$)',line): found.append(re.sub(r'\s+',' ',line))
    return found


def torch_ops(text):
    count=0
    for raw in text.splitlines():
        line=raw.strip()
        if not line or line.startswith(('module','func.func','^bb','}','{')): continue
        if '=' in line or re.search(r'\b(?:torch|arith|tensor|func|prim)\.',line): count+=1
    return count


def archived_compiler_metrics(root):
    root=Path(root); checks=[]; sources={}
    def check(name,actual,expected):
        checks.append({'id':name,'computed':actual,'expected':expected,'passed':actual==expected})
    for dialect in ('linalg','torch'):
        rows,path=scan_rows(root,dialect); sources[str(path.relative_to(root))]=sha(path.read_bytes())
        if dialect=='linalg':
            summary={'programs':len(rows),'baseline_success':sum(truth(r['baseline_success']) for r in rows),
                'fusion_success':sum(truth(r['fusion_success']) for r in rows),
                'fusion_changed_ir':sum(truth(r['fusion_changed_ir']) for r in rows)}
            summary['unchanged']=sum(truth(r['fusion_success']) and not truth(r['fusion_changed_ir']) for r in rows)
            summary['baseline_failures']=len(rows)-summary['baseline_success']
            summary['fusion_failures_after_baseline']=sum(truth(r['baseline_success']) and not truth(r['fusion_success']) for r in rows)
            expected=json.loads((path.parent/'fusion_scan_summary.json').read_text())
            for key,value in expected.items(): check('linalg.'+key,summary[key],value)
            check('paper.linalg_valid',summary['fusion_success'],693)
            check('paper.linalg_changed_percent',round(100*summary['fusion_changed_ir']/summary['fusion_success'],1),91.8)
        else:
            for name in PASSES:
                success=[r for r in rows if truth(r.get(name+'_success') or 'False')]
                check('torch.'+name+'.success',len(success),687)
                if name=='simplification_pipeline':
                    changed=sum(truth(r[name+'_changed_ir']) for r in success)
                    check('paper.torch_changed_percent',round(100*changed/len(success),1),45.0)
    path=root/'data/downstream/records/DT1_ELEMENTWISE/outputs/riscv_backend_extension/riscv_backend_results.csv'
    rows=read_csv(path); sources[str(path.relative_to(root))]=sha(path.read_bytes())
    if len({r['dt1_id'] for r in rows})!=len(rows): raise ValueError('Duplicate RISC-V result ID')
    both=[r for r in rows if truth(r['both_compile'])]
    check('riscv.programs',len(rows),693);check('riscv.both_compile',len(both),675)
    for row in both:
        before,after=int(float(row['baseline_instruction_count'])),int(float(row['fused_instruction_count']))
        outcome='IMPROVED' if after<before else 'TIED' if after==before else 'WORSENED'
        check('riscv.outcome.'+row['dt1_id'],outcome,row['instruction_outcome'])
        check('riscv.removed.'+row['dt1_id'],before-after,int(float(row['instructions_removed'])))
    rates={}
    for benefit,expected in [('LOW',1.7),('MEDIUM',31.6),('HIGH',24.9)]:
        group=[r for r in both if r['fusion_benefit']==benefit]
        improved=sum(int(float(r['fused_instruction_count']))<int(float(r['baseline_instruction_count'])) for r in group)
        rates[benefit]={'programs':len(group),'improved':improved,'percent':100*improved/len(group)}
        check('paper.riscv_improvement.'+benefit,round(rates[benefit]['percent'],1),expected)
    return {'mode':'archived_compiler_observation_recalculation','passed':all(c['passed'] for c in checks),
            'checks':checks,'sources':sources,'riscv_improvement':rates,
            'fresh_compilation':False,'note':'Recalculates from recorded compiler counts; does not substitute for fresh transformations/code generation.'}


def run_command(command,timeout):
    start=time.monotonic()
    try:
        p=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
        return {'command':command,'returncode':p.returncode,'success':p.returncode==0,
                'stdout':p.stdout,'stderr':p.stderr,'seconds':time.monotonic()-start,'timeout':False}
    except subprocess.TimeoutExpired:
        return {'command':command,'returncode':None,'success':False,'stdout':'','stderr':'Command timed out',
                'seconds':time.monotonic()-start,'timeout':True}


def toolchain(bin_dir=None):
    prefix=Path(bin_dir or '/opt/paper-backend/bin'); result={}
    for name in ('iree-opt','iree-compile'):
        path=prefix/name
        if not path.is_file(): raise FileNotFoundError('Missing pinned paper compiler: '+str(path))
        version=run_command([str(path),'--version'],30)
        if not version['success']: raise RuntimeError('Unable to inspect paper compiler')
        result[name]={'path':str(path),'version':version['stdout']+version['stderr'],
                      'launcher_sha256':sha(path.read_bytes())}
    # iree-opt reports LLVM only; the companion compile identifies the package build.
    if BACKEND_VERSION not in result['iree-compile']['version']:
        raise ValueError('Paper backend version differs from '+BACKEND_VERSION)
    return result


def fresh_compiler_experiments(root,output,*,dialect='linalg',limit=None,workers=2,timeout=180,bin_dir=None,riscv=False):
    if dialect not in ('linalg','torch') or (riscv and dialect!='linalg'): raise ValueError('Invalid compiler experiment dialect')
    if workers<1 or workers>16 or timeout<=0 or (limit is not None and limit<1): raise ValueError('Invalid resource limit')
    root,output=Path(root),Path(output)
    if output.exists(): raise FileExistsError('Choose a new compiler output directory')
    tools=toolchain(bin_dir); rows,scan_path=scan_rows(root,dialect); lookup=source_lookup(root)
    if any(r['mlir_hash'] not in lookup for r in rows): raise ValueError('Unresolved immutable program hash')
    all_rows=sorted(rows,key=lambda r:r['dt1_id']); selected=all_rows[:limit] if limit else all_rows
    if riscv: selected=[r for r in selected if truth(r['fusion_success'])]
    output.mkdir(parents=True); records_dir=output/'records';records_dir.mkdir()
    def process(row):
        folder=output/'programs'/row['dt1_id'];folder.mkdir(parents=True)
        source=folder/'source.mlir';source.write_text(lookup[row['mlir_hash']]+'\n')
        opt=tools['iree-opt']['path']; compile_path=tools['iree-compile']['path']
        extra=['--mlir-print-op-generic'] if dialect=='torch' else []
        baseline=run_command([opt,str(source),*extra],timeout)
        result={'dt1_id':row['dt1_id'],'mlir_hash':row['mlir_hash'],'dialect':dialect,'baseline':baseline,'passes':{}}
        if baseline['success']:
            experiments={'fusion':['--linalg-fuse-elementwise-ops']} if dialect=='linalg' else PASSES
            for name,flags in experiments.items():
                transformed=run_command([opt,str(source),*extra,*flags],timeout)
                result['passes'][name]=transformed
                if transformed['success']:
                    before,after=baseline['stdout'],transformed['stdout']
                    normalized=lambda s:re.sub(r'\s+',' ',s).strip()
                    transformed['changed_ir']=(before!=after if dialect=='linalg' else normalized(before)!=normalized(after))
                    counter=(lambda s:len(re.findall(r'\blinalg\.[A-Za-z0-9_]+',s))) if dialect=='linalg' else torch_ops
                    transformed['ops_before']=counter(before);transformed['ops_after']=counter(after)
                    transformed['ops_removed']=counter(before)-counter(after)
            if riscv and result['passes']['fusion']['success']:
                result['riscv']={}
                for label,text in [('baseline',baseline['stdout']),('fused',result['passes']['fusion']['stdout'])]:
                    sub=folder/label;sub.mkdir();ir=sub/'input.mlir';ir.write_text(text)
                    command=[compile_path,str(ir),'-o',str(sub/'program.vmfb'),'--iree-hal-target-device=local',
                        '--iree-hal-local-target-device-backends=llvm-cpu','--iree-llvmcpu-target-triple=riscv64',
                        '--iree-llvmcpu-target-abi=lp64d','--iree-opt-level=O3',
                        '--iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+zvl512b,+v',
                        '--iree-hal-dump-executable-files-to='+str(sub)]
                    codegen=run_command(command,timeout);assembly=[]
                    for p in sorted(sub.rglob('*.s')): assembly.extend(instructions(p.read_text(errors='replace')))
                    codegen.update(instruction_count=len(assembly),assembly_files=len(list(sub.rglob('*.s'))),
                                   instruction_stream_sha256=sha('\n'.join(assembly).encode()) if assembly else None)
                    result['riscv'][label]=codegen
        (records_dir/(row['dt1_id']+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool: records=list(pool.map(process,selected))
    mismatches=[]
    for row,actual in zip(selected,records):
        pairs=[('baseline_success',actual['baseline']['success'],truth(row['baseline_success']))]
        if dialect=='linalg':
            got=actual['passes'].get('fusion',{});pairs += [('fusion_success',got.get('success',False),truth(row['fusion_success']))]
            if got.get('success'):
                pairs += [('fusion_changed_ir',got['changed_ir'],truth(row['fusion_changed_ir'])),
                          ('linalg_ops_before',got['ops_before'],int(float(row['linalg_ops_before']))),
                          ('linalg_ops_after',got['ops_after'],int(float(row['linalg_ops_after'])))]
        else:
            for name in PASSES:
                got=actual['passes'].get(name,{})
                pairs.append((name+'_success',got.get('success',False),truth(row.get(name+'_success') or 'False')))
                if got.get('success'):pairs += [(name+'_changed_ir',got['changed_ir'],truth(row[name+'_changed_ir'])),(name+'_ops_removed',got['ops_removed'],int(float(row[name+'_ops_removed'])))]
        for field,got,expected in pairs:
            if got!=expected:mismatches.append({'dt1_id':row['dt1_id'],'field':field,'observed':got,'archived':expected})
    summary={'mode':'fresh_compiler_execution','dialect':dialect,'full_corpus':limit is None,'programs':len(records),
        'expected_programs':len(all_rows),'selection':'archived fusion-valid subset' if riscv else 'all archived scan rows',
        'driver_sha256':sha(Path(__file__).read_bytes()),'toolchain':tools,'source_scan_sha256':sha(scan_path.read_bytes()),
        'mismatches':mismatches,'transformation_agreement':not mismatches,'riscv_requested':riscv,
        'note':'Compile failures and differences are evidence, never silently replaced by archived results.'}
    if riscv:
        historical=read_csv(root/'data/downstream/records/DT1_ELEMENTWISE/outputs/riscv_backend_extension/riscv_backend_results.csv')
        by_id={r['dt1_id']:r for r in historical};rv_differences=[]
        for actual in records:
            pair=actual.get('riscv',{});both=all(pair.get(k,{}).get('success') and pair[k].get('assembly_files',0)>0 for k in ('baseline','fused'))
            expected=by_id[actual['dt1_id']]
            if both!=truth(expected['both_compile']):rv_differences.append({'dt1_id':actual['dt1_id'],'field':'both_compile'})
            if both:
                for label in ('baseline','fused'):
                    observed=pair[label]['instruction_count'];old=int(float(expected[label+'_instruction_count']))
                    if observed!=old:rv_differences.append({'dt1_id':actual['dt1_id'],'field':label+'_instruction_count','observed':observed,'archived':old})
                b,a=pair['baseline']['instruction_count'],pair['fused']['instruction_count']
                outcome='IMPROVED' if a<b else 'TIED' if a==b else 'WORSENED'
                if outcome!=expected['instruction_outcome']:rv_differences.append({'dt1_id':actual['dt1_id'],'field':'instruction_outcome','observed':outcome,'archived':expected['instruction_outcome']})
        summary['riscv_mismatches']=rv_differences
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
