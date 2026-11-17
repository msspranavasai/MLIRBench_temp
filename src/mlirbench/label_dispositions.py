"""Verify the non-mutating case-disposition dossier and replay its saved witnesses."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from .evaluation.reproduce import LABELS


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def load_dossier(root):
    root=Path(root)
    manifest=json.loads((root/'experiments/label_review_manifest.json').read_text())
    if manifest.get('schema')!='mlirbench-label-review-manifest-v1':
        raise ValueError('Unknown label-review manifest')
    name=manifest['dossier']
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path.read_bytes())!=manifest['dossier_sha256']:
        raise ValueError('Label-disposition dossier checksum mismatch')
    ledger=json.loads(path.read_text())
    if ledger.get('schema')!='mlirbench-label-dispositions-v1':raise ValueError('Unknown disposition schema')
    master_path=root/'data/processed/MLIRBench/master/master.jsonl'
    expected=manifest['published_master_sha256']
    if sha(master_path.read_bytes())!=expected or ledger['scope']['published_master_sha256']!=expected:
        raise ValueError('Published dataset differs from the reviewed snapshot')
    master={r['program_id']:r for r in map(json.loads,master_path.read_text().splitlines())}
    cases=ledger['cases'];ids=[c['program_id'] for c in cases]
    if len(ids)!=len(set(ids)) or sorted(ids)!=manifest['case_ids'] or len(ids)!=ledger['scope']['cases']:
        raise ValueError('Disposition coverage differs from the reviewed case inventory')
    for case in cases:
        record=master[case['program_id']]
        if case['source_sha256']!=[sha(record[k].encode()) for k in ('input_mlir','mutated_mlir')]:
            raise ValueError('Disposition program hash mismatch: '+case['program_id'])
        if case['published_label']!=LABELS[str(record['judge__outputs_diverge'])]:
            raise ValueError('Disposition does not preserve the published label: '+case['program_id'])
        if case.get('published_record_modified') is not False or case.get('historical_runtime_inputs_proven') is not False:
            raise ValueError('Dossier cannot claim a record change or recovered historical runtime')
        if digest(case['witness_inputs'])!=case['witness_input_sha256']:
            raise ValueError('Saved witness input checksum mismatch')
        extra=case.get('additional_numeric_witness')
        if extra and digest(extra['inputs'])!=extra['input_sha256']:raise ValueError('Additional witness checksum mismatch')
    if dict(Counter(c['documented_disposition'] for c in cases))!=ledger['disposition_counts']:
        raise ValueError('Disposition summary differs from case records')
    current={name:sha((root/'src/mlirbench/execution'/name).read_bytes()) for name in manifest['reviewed_execution_code_sha256']}
    return ledger,master,current==manifest['reviewed_execution_code_sha256']


def numeric_equal(a,b):
    """Diagnostic paper-named value comparison; not a replacement for typed-exact."""
    import numpy as np
    if a['kind']!=b['kind']:return False
    if a['kind']=='sequence':
        return len(a['items'])==len(b['items']) and all(numeric_equal(x,y) for x,y in zip(a['items'],b['items']))
    decode=lambda x:np.frombuffer(base64.b64decode(x['bytes_b64']),dtype=x['dtype']).reshape(x['shape'])
    return bool(np.array_equal(decode(a),decode(b)))


def iterator_control(source):
    old='linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]}'
    new='linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["reduction", "parallel"]}'
    if source.count(old)!=1:raise ValueError('Diagnostic iterator control no longer matches the reviewed source')
    return source.replace(old,new)


def review_labels(root,output,*,fresh=False,ids=None,all_cases=False,repeats=10,numeric_witness=False,control=False):
    root,output=Path(root),Path(output)
    if output.resolve().is_relative_to((root/'data').resolve()):raise ValueError('Review outputs must stay outside scientific data directories')
    if output.exists():raise FileExistsError('Choose a new review output directory')
    if not 2<=repeats<=100:raise ValueError('Use 2 to 100 fresh-process repetitions')
    ledger,master,core_matches=load_dossier(root)
    selected=ledger['cases']
    if ids:
        if set(ids)-{c['program_id'] for c in selected}:raise ValueError('ID is not in the reviewed disposition inventory')
        selected=[c for c in selected if c['program_id'] in ids]
    elif fresh and not all_cases:
        selected=[c for c in selected if c['program_id']=='seed1_linalg_prog10_L2_R4']
    if not selected:raise ValueError('No reviewed cases selected')
    if fresh and not core_matches:raise ValueError('Execution code differs from this review; create a separately identified investigation')
    if (numeric_witness or control) and not fresh:raise ValueError('Witness/control options require --fresh')
    if numeric_witness and any('additional_numeric_witness' not in c for c in selected):
        raise ValueError('Additional numeric witness is unavailable for one or more selected cases')
    if control and (len(selected)!=1 or selected[0]['program_id']!='seed1_linalg_prog10_L2_R4'):
        raise ValueError('Iterator control requires only seed1_linalg_prog10_L2_R4')
    output.mkdir(parents=True)
    findings=[]
    for case in selected:
        item={'program_id':case['program_id'],'published_label':case['published_label'],
              'original_assessment':case['original_assessment'],'documented_disposition':case['documented_disposition'],
              'diagnosis':case['diagnosis'],'action':case['action']}
        if fresh:
            from .execution.pair_validation import validate_pair
            folder=output/case['program_id'];folder.mkdir()
            record=master[case['program_id']]
            inputs=case['additional_numeric_witness']['inputs'] if numeric_witness else case['witness_inputs']
            (folder/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
            for field,name in [('input_mlir','original.mlir'),('mutated_mlir','mutated.mlir')]:
                (folder/name).write_text(record[field])
            verifier={}
            for side in ('original','mutated'):
                standard='mlir-opt' if case['dialect']=='linalg' else 'torch-mlir-opt'
                for label,tool in [('standalone',standard),('paper_backend','/opt/paper-backend/bin/iree-opt')]:
                    try:
                        p=subprocess.run([tool,'--verify-each',str(folder/(side+'.mlir')),'-o','/dev/null'],capture_output=True,text=True,timeout=60)
                        verifier[side+'_'+label]={'returncode':p.returncode,'stderr':p.stderr}
                    except (OSError,subprocess.TimeoutExpired) as exc:
                        verifier[side+'_'+label]={'error':str(exc)}
            validation=validate_pair(record['input_mlir'],record['mutated_mlir'],case['dialect'],recorded_inputs=inputs,repeats=repeats,timeout=60)
            for trial in validation['trials']:
                runs=trial.get('runs',[])
                if len(runs)==2 and all(r.get('status')=='success' for side in runs for r in side):
                    trial['diagnostic_numpy_array_equal']=numeric_equal(runs[0][0]['output_typed'],runs[1][0]['output_typed'])
            raw={'source_verifiers':verifier,'validation':validation}
            if control:
                diagnostic=iterator_control(record['mutated_mlir'])
                (folder/'diagnostic-iterator-control.mlir').write_text(diagnostic)
                raw['iterator_control']={'diagnostic_copy_only':True,'published_programs_modified':False,
                    'source_sha256':sha(diagnostic.encode()),'validation':validate_pair(record['input_mlir'],diagnostic,case['dialect'],
                    recorded_inputs=inputs,repeats=repeats,timeout=60)}
            (folder/'replay.json').write_text(json.dumps(raw,indent=2,allow_nan=False)+'\n')
            item.update(fresh_verdict=validation['verdict'],replay=str(folder.name+'/replay.json'))
            print(case['program_id']+': '+validation['verdict'],flush=True)
        findings.append(item)
    report={'schema':'mlirbench-disposition-review-v1','status':'DOCUMENTED_WITH_LIMITATIONS',
            'cases':len(findings),'fresh_experiments':fresh,'execution_code_matches_recorded_review':core_matches,
            'historical_runtime_inputs_proven':False,'published_dataset_modified':False,'paid_api_calls_requested':False,
            'disposition_counts':dict(Counter(c['documented_disposition'] for c in findings)),
            'scientific_limits':ledger['scientific_limits'],'findings':findings}
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Label-conflict disposition review','','**DOCUMENTED_WITH_LIMITATIONS**','',
           'This is a review overlay. Published records and paper calculations are unchanged.', '',
           '| Program | Published | Documented disposition |','|---|---|---|']
    lines += ['| '+c['program_id']+' | '+c['published_label']+' | '+c['documented_disposition']+' |' for c in findings]
    lines += ['', 'Historical runtime inputs remain unproven. Fresh replay does not overwrite a disposition or establish universal equivalence.',
              'Nondeterminism frequency is not an acceptance target; another batch can observe a different frequency or no variation.',
              'See docs/label-dispositions.md for the recorded investigation and diagnostic controls.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report
