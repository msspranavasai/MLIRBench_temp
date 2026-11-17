"""Reviewer pipeline: explicit archived replay, real CPU execution and opt-in APIs."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from .dataset import ensure_data, HF_REVISION
from .stage_artifacts import STAGES, digest, pair_digest, record, save_records, load_run, atomic_json

VIEWS = ('master/master.jsonl','reasoning/reasoning.jsonl','evaluation/evaluation.jsonl')
LABELS = {'True':'DIVERGE','False':'EQUIV','N/A':'ERROR'}


def rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def score_components(predicted, ground, result, explanation):
    if predicted not in LABELS.values() or ground not in LABELS.values():
        raise ValueError('Unscorable label; unsupported cases cannot receive an invented score')
    result, explanation = float(result), float(explanation)
    if not (math.isfinite(result) and 0 <= result <= 10 and math.isfinite(explanation) and 0 <= explanation <= 4):
        raise ValueError('Invalid judge component score')
    hard = int(predicted == ground)
    # Same floating-point order as the original judge; retain the unrounded total too.
    total = hard/3*100 + result/10/3*100 + explanation/4/3*100
    return {'ground_truth_label':ground,'predicted_label':predicted,
            'hard_label_accuracy':float(hard),'result_accuracy':result,'explanation_rating':explanation,
            'weighted_hard_label_score':round(hard/3*100,2),
            'weighted_result_score':round(result/10/3*100,2),
            'weighted_explanation_score':round(explanation/4/3*100,2),
            'total_score':round(total,2),'unrounded_total_score':total}


def execution_data(report):
    """EQUIV is an observation on the declared suite, never a universal proof."""
    verdict = report['verdict']
    label = {'NO_DIFFERENCE_OBSERVED':'EQUIV','DIVERGE':'DIVERGE',
             'COMPILE_ERROR':'ERROR','RUNTIME_ERROR':'ERROR'}.get(verdict)
    result = {'ground_truth_label':label,'validation_verdict':verdict,
              'label_scope':'declared_valid_input_suite' if label in ('EQUIV','DIVERGE') else 'current_toolchain_error' if label else 'unverified',
              'validation_report':report,'historical_runtime_inputs_proven':False}
    if label in ('EQUIV','DIVERGE'):
        from .execution.pair_ground_truth import _decode
        t = next(t for t in report['trials'] if t['status']==verdict)
        result.update(embedded_inputs={n:x['data'] for n,x in zip(report['argument_mapping'][0],t['inputs'])},
                      programA_output=_decode(t['runs'][0][0]['output_typed']),
                      programB_output=_decode(t['runs'][1][0]['output_typed']),
                      compilation_status_A='success',compilation_status_B='success',
                      outputs_diverge=label=='DIVERGE')
    else:
        comp = report.get('compilation',[{},{}])
        result.update(compilation_status_A=comp[0].get('status'),compilation_status_B=comp[1].get('status'),
                      outputs_diverge=None,programA_output=None,programB_output=None,embedded_inputs={})
    return result


def replay_execution(pair, historical):
    fields = historical['historical_fields']
    def parsed(name):
        try: return json.loads(fields.get(name) or 'null')
        except (ValueError,TypeError): return None
    return {'ground_truth_label':LABELS[str(pair['judge__outputs_diverge'])],
            'label_scope':'published_historical_observation',
            'historical_runtime_inputs_proven':False,
            'embedded_inputs':parsed('judge__embedded_inputs_json'),
            'programA_output':parsed('judge__programA_output_json'),
            'programB_output':parsed('judge__programB_output_json'),
            'compilation_status_A':fields.get('judge__compilation_status_A'),
            'compilation_status_B':fields.get('judge__compilation_status_B'),
            'outputs_diverge':None if str(pair['judge__outputs_diverge'])=='N/A' else str(pair['judge__outputs_diverge'])=='True',
            'historical_record':historical,
            'note':'Archived observations, not a fresh execution. Embedded inputs are unproven runtime metadata.'}


def require_replay_parent(parent):
    if parent['mode'] != 'replay':
        raise ValueError('Cannot attach archived downstream observations to a fresh upstream stage. Rerun downstream stages, or start a separate archived replay.')


def verify_archived_executions(root, records, pairs, verified=None):
    """Bind replayed checkpoints to frozen observations, including on resume."""
    replayed = [r for r in records if r['mode'] == 'replay']
    if not replayed:
        return None
    if verified is None:
        from .execution.provenance import load_verified_sources
        verified = load_verified_sources(root)
    master, historical, verification = verified
    expected_path = 'data/historical/' + verification['sidecar_filename']
    for item in replayed:
        pid = item['program_id']
        if pid not in master or pairs.get(pid) != master[pid]:
            raise ValueError(f'Archived execution program differs from frozen source: {pid}')
        source = item['source']
        if (source.get('path') != expected_path
                or source.get('file_sha256') != verification['sidecar_sha256']):
            raise ValueError(f'Archived execution source hash mismatch: {pid}')
        expected = replay_execution(master[pid], historical[pid])
        if any(k not in item['data'] or item['data'][k] != value
               for k, value in expected.items() if k != 'note'):
            raise ValueError(f'Archived execution observation differs from frozen source: {pid}')
    return verification


def missing_judge_observations(artifacts):
    """Derive missing cases from selected records, including reused checkpoints.

    No judge stage means judging has not yet been requested, not missing data.
    An explicit unscored judge record is an observation and is not fabricated
    into a scored result or counted as absent. Historical manifest counts are
    not trusted: older resumes can have lost them or retained unrelated cases.
    """
    if 'judge' not in artifacts:
        return []
    observed = {(r['program_id'], r['model']) for r in artifacts['judge']}
    return [{'program_id': r['program_id'], 'model': r['model'],
             'reason': 'No judge observation for selected reasoning record'}
            for r in artifacts.get('reasoning', [])
            if (r['program_id'], r['model']) not in observed]


def run_pipeline(root, output, **kwargs):
    output = Path(output)
    if output.exists(): raise FileExistsError(f'Choose a new output directory: {output}')
    try:
        return _run_pipeline(root,output,**kwargs)
    except BaseException as exc:
        path = output/'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            manifest.update(status='FAILED' if isinstance(exc, Exception) else 'INTERRUPTED',failure_type=type(exc).__name__)
            atomic_json(path, manifest)
        raise


def _run_pipeline(root, output, *, stage='all', mode='replay', ids=None, models=None,
                 input_run=None, all_records=False, allow_paid=False, source=None,
                 dialect=None, level=None, requirement=None, resume_incomplete=False):
    root, output = Path(root), Path(output)
    if output.exists(): raise FileExistsError(f'Choose a new output directory: {output}')
    if mode not in ('replay','fresh'): raise ValueError('Unknown pipeline mode')
    if stage not in (*STAGES,'metrics','all'): raise ValueError('Unknown stage')
    if mode=='fresh' and stage=='metrics': raise ValueError('Metrics are deterministic; use --stage metrics --input without --mode fresh')
    if mode == 'fresh' and stage in ('mutation','reasoning','judge','all') and not allow_paid:
        raise PermissionError('Model calls are disabled. Default replay costs nothing. Fresh model stages require --allow-paid-api and an explicitly selected model where applicable.')
    if mode == 'fresh' and stage in ('reasoning','judge','all') and (not models or len(models)!=1):
        raise ValueError('Choose exactly one reasoning --model for a fresh run; larger experiments must be invoked explicitly.')
    live_configuration = None
    if mode == 'fresh' and stage in ('mutation','reasoning','judge','all'):
        from .live import preflight
        live_configuration = preflight(root, ('mutation','reasoning','judge') if stage == 'all' else (stage,), models)
    if resume_incomplete and (mode != 'fresh' or not input_run or source or stage == 'metrics'):
        raise ValueError('--resume-incomplete requires a stopped fresh run via --input, the same stage, and no --source')
    data = root/'data/processed/MLIRBench'
    inherited, input_manifest = {}, None
    if input_run:
        inherited,input_manifest = load_run(input_run)
        if 'mutation' not in inherited and not resume_incomplete: raise ValueError('Input run has no mutation stage')
    old_plan, old_cases = None, None
    if resume_incomplete:
        from .case_journal import read_plan, load_journal
        if input_manifest.get('status') not in ('FAILED','INTERRUPTED','IN_PROGRESS','STAGES_COMPLETE'):
            raise ValueError('--resume-incomplete requires a stopped incomplete run; completed stages can be rerun without this flag')
        old_plan = read_plan(input_run, input_manifest)
        hashes = {r['record_sha256']:r for values in inherited.values() for r in values}
        old_cases = load_journal(input_run, input_manifest, by_hash=hashes)
    end = len(STAGES) if stage in ('all','metrics') else STAGES.index(stage)+1
    target = 0 if stage=='all' else len(STAGES) if stage=='metrics' else STAGES.index(stage)
    required, replayed = ({VIEWS[0]} if not source and not input_run else set()), []
    for index, name in enumerate(STAGES[:end]):
        reuse = name in inherited and (index < target or resume_incomplete)
        if stage == 'metrics' and input_run and name not in inherited:
            raise ValueError(f'Metrics input is missing {name} artifacts')
        if reuse:
            if name == 'execution' and any(r['mode']=='replay' for r in inherited[name]):
                required.add(VIEWS[0])  # Frozen-source verification still applies.
        elif not (source and name == 'mutation') and (mode == 'replay' or index < target):
            replayed.append(name)
            required.add({'mutation':VIEWS[0],'execution':VIEWS[0],
                          'reasoning':VIEWS[1],'judge':VIEWS[2]}[name])
    for name in replayed:
        for parent_name in {'mutation':(),'execution':('mutation',),
                            'reasoning':('execution',),'judge':('execution','reasoning')}[name]:
            for parent in inherited.get(parent_name, []): require_replay_parent(parent)
    # All-archive metrics compare coverage against published views. Fresh or
    # mixed checkpoints only produce scoped metrics and need no such comparison.
    if stage in ('all','metrics') and not source and mode == 'replay':
        if not input_run or all(r['mode']=='replay' for values in inherited.values() for r in values):
            required.update(VIEWS)
    resolution = ensure_data(root, ['MLIRBench/'+n for n in VIEWS if n in required]) if required else {
        'files':{}, 'note':'self-contained fresh inputs/checkpoint; no published data required'}
    published = {r['program_id']:r for r in rows(data/VIEWS[0])} if VIEWS[0] in required else {}
    available = {r['program_id']:r['data'] for r in inherited.get('mutation', [])} if input_run else published
    if resume_incomplete:
        available = {r['program_id']:r for r in old_plan['selected_pairs']}
    selected_ids = set(ids or (available if mode=='replay' or all_records else ([sorted(available)[0]] if available else [])))
    if source:
        if mode != 'fresh' or stage not in ('mutation','all') or input_run or ids:
            raise ValueError('--source requires a fresh mutation/all run without --input or --id')
        if dialect not in ('linalg','torch') or level not in range(1,6):
            raise ValueError('A custom source requires --dialect and --level 1..5')
        text = Path(source).read_text()
        pid = 'custom_'+hashlib.sha256(text.encode()).hexdigest()[:16]
        available = {pid:{'program_id':pid,'input_mlir':text,'dialect':dialect,'level':level,
                           'requirement_id':requirement,'seed':0}}
        selected_ids = {pid}
    if selected_ids - available.keys(): raise ValueError(f'Unknown IDs: {sorted(selected_ids-available.keys())}')
    selected = [available[k] for k in sorted(selected_ids)]
    if mode=='fresh' and not all_records and len(selected)>1:
        raise ValueError('Fresh runs default to one pair; use --all for an explicitly requested multi-pair run')
    from .ae import source_fingerprint
    fingerprint = source_fingerprint(root)
    case_plan = None
    if mode == 'fresh':
        case_plan = {'schema':'mlirbench-case-plan-v1', 'requested_stage':stage,
                     'models':models or [], 'selected_pairs':selected,
                     'source_fingerprint':fingerprint, 'live_configuration':live_configuration}
        if resume_incomplete and case_plan != old_plan:
            raise ValueError('Incomplete-call resume requires unchanged stage, selected inputs, models, code and live configuration')
    artifacts, sources = {}, {}
    def origin(path):
        path = Path(path)
        key = str(path.relative_to(root))
        if key not in sources: sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'kind':'published_archive','path':key,'file_sha256':sources[key],
                'hf_revision':HF_REVISION if key.startswith('data/processed/') else None,
                'lineage':'reconstructed_from_published_records; original end-to-end runtime lineage unproven'}
    output.mkdir(parents=True)
    manifest = {'schema':'mlirbench-pipeline-run-v1','requested_stage':stage,'requested_mode':mode,
                'status':'IN_PROGRESS','artifacts':{},'dataset_resolution':resolution,'source_files':sources,
                'historical_end_to_end_runtime_proven':False,
                'score_formula':'100/3 * (label correctness + result_accuracy/10 + explanation_rating/4)',
                'missing_judge_observations':[], 'reused_stages':[],
                'input_run': {'manifest_sha256':hashlib.sha256((Path(input_run)/'manifest.json').read_bytes()).hexdigest(),
                              'source_fingerprint':input_manifest.get('source_fingerprint')} if input_run else None}
    if live_configuration is not None:
        manifest['live_configuration'] = live_configuration
    manifest['source_fingerprint'] = fingerprint
    journal = None
    if case_plan is not None:
        from .case_journal import CaseJournal
        journal = CaseJournal(output, case_plan, old_cases)
        manifest['case_plan'] = journal.plan_spec()
        manifest['resumed_incomplete'] = resume_incomplete
    def checkpoint(name, values):
        manifest['artifacts'][name] = save_records(output/(name+'.jsonl'),values)
        atomic_json(output/'manifest.json', manifest)
    for index, name in enumerate(STAGES[:end]):
        manifest['active_stage'] = name
        atomic_json(output/'manifest.json', manifest)
        # Ordinary --input reruns the target. Explicit continuation also reuses
        # completed fresh stages; case journals recover partial stage successes.
        if name in inherited and (index < target or resume_incomplete):
            artifacts[name] = list(inherited[name]) if resume_incomplete else [r for r in inherited[name] if r['program_id'] in selected_ids and (not models or not r['model'] or r['model'] in models)]
            if name == 'execution':
                verification = verify_archived_executions(root, artifacts[name], {r['program_id']:r['data'] for r in artifacts['mutation']})
                if verification is not None: manifest['historical_source_verification'] = verification
            manifest['reused_stages'].append(name)
            checkpoint(name,artifacts[name])
            continue
        actual_mode = mode if index >= target else 'replay'
        if stage=='metrics':
            if name not in inherited and input_run: raise ValueError(f'Metrics input is missing {name} artifacts')
            if name in inherited:
                artifacts[name] = list(inherited[name]) if resume_incomplete else [r for r in inherited[name] if r['program_id'] in selected_ids and (not models or not r['model'] or r['model'] in models)]
                manifest['reused_stages'].append(name)
                checkpoint(name,artifacts[name])
                continue
            actual_mode = 'replay'
        if name=='mutation':
            values = []
            for pair in selected:
                if actual_mode=='fresh':
                    from .live import mutate
                    def generate():
                        new = mutate(root,pair)
                        new['comparison_program_id'] = pair['program_id']
                        new['program_id'] = 'fresh_'+pair_digest(new)[:24]
                        return record(name,new,new,mode='fresh',source={'kind':'live_model','comparison_program_id':pair['program_id']})
                    values.append(journal.call(name, pair, generate))
                else: values.append(record(name,pair,pair,mode='replay',source=origin(data/VIEWS[0])))
            artifacts[name] = values
            selected = [r['data'] for r in values]
        elif name=='execution':
            if actual_mode == 'replay':
                from .execution.provenance import load_verified_sources
                verified = load_verified_sources(root)
                _, historical, verification = verified
                manifest['historical_source_verification'] = verification
            values = []
            for parent in artifacts['mutation']:
                pair = parent['data']
                if actual_mode=='replay':
                    require_replay_parent(parent)
                    payload = replay_execution(pair,historical[pair['program_id']])
                    source_info = origin(root/'data/historical'/verification['sidecar_filename'])
                else:
                    from .execution.pair_validation import validate_pair
                    values.append(journal.call(name, pair, lambda: record(name, pair,
                        execution_data(validate_pair(pair['input_mlir'],pair['mutated_mlir'],pair['dialect'])),
                        mode='fresh', parents=[parent['record_sha256']],
                        source={'kind':'real_cpu_execution','comparison':'typed-exact-value-v1'}),
                        parents=[parent['record_sha256']]))
                    continue
                values.append(record(name,pair,payload,mode=actual_mode,parents=[parent['record_sha256']],source=source_info))
            if actual_mode == 'replay':
                verify_archived_executions(root, values, {r['program_id']:r['data'] for r in artifacts['mutation']}, verified=verified)
            artifacts[name] = values
        elif name=='reasoning':
            archived = rows(data/VIEWS[1]) if actual_mode=='replay' else []
            by_id = {r['program_id']:r for r in artifacts['execution']}
            pairs = {r['program_id']:r['data'] for r in artifacts['mutation']}
            values = []
            if actual_mode=='fresh':
                from .live import reason
                for pid,parent in by_id.items():
                    values.append(journal.call(name, pairs[pid], lambda: record(name, pairs[pid],
                        reason(root,pairs[pid],parent['data'],models[0]), mode='fresh',
                        parents=[parent['record_sha256']],source={'kind':'live_model'},model=models[0]),
                        parents=[parent['record_sha256']], model=models[0]))
            else:
                for item in archived:
                    pid,model = item['program_id'],item['model_key']
                    if pid not in by_id or models and model not in models: continue
                    parent = by_id[pid]; require_replay_parent(parent)
                    if pair_digest(item) != pair_digest(pairs[pid]): raise ValueError('Archived response belongs to a different program pair')
                    payload = {k:v for k,v in item.items() if k not in ('input_mlir','mutated_mlir','judge__outputs_diverge','judge__semantic_explanation')}
                    if payload.get('predicted_label') is None:
                        payload.update(archived_predicted_label=None,predicted_label='ERROR',
                                       normalization='original _row_for_model failure-to-ERROR rule; raw null preserved')
                    values.append(record(name,pairs[pid],payload,mode='replay',parents=[parent['record_sha256']],source={**origin(data/VIEWS[1]),'row_sha256':digest(item)},model=model))
            if not values: raise ValueError('No reasoning records for the requested IDs/models')
            artifacts[name] = values
        elif name=='judge':
            archived = {(r['program_id'],r['evaluated_model']):r for r in rows(data/VIEWS[2])} if actual_mode=='replay' else {}
            executions = {r['program_id']:r for r in artifacts['execution']}
            pairs = {r['program_id']:r['data'] for r in artifacts['mutation']}
            values = []
            for prediction in artifacts['reasoning']:
                pid,model = prediction['program_id'],prediction['model']
                execution = executions[pid]
                if actual_mode=='replay':
                    require_replay_parent(prediction); require_replay_parent(execution)
                    raw = archived.get((pid,model))
                    if raw is None:
                        continue
                    if raw['predicted_label'] != prediction['data']['predicted_label'] or raw['ground_truth_label'] != execution['data']['ground_truth_label']:
                        raise ValueError(f'Archived judge label does not match its selected parents: {pid}/{model}')
                    payload = score_components(raw['predicted_label'],raw['ground_truth_label'],raw['result_accuracy'],raw['explanation_rating'])
                    payload.update(program_id=pid,evaluated_model=model,verdict=raw['verdict'],
                                   archived_total_score=raw['total_score'],archived_score_record=raw,
                                   score_delta=round(payload['total_score']-float(raw['total_score']),8),
                                   subjective_scores='archived observations; not regenerated')
                    source_info = {**origin(data/VIEWS[2]),'row_sha256':digest(raw)}
                else:
                    from .live import judge
                    parents = [execution['record_sha256'],prediction['record_sha256']]
                    values.append(journal.call(name, pairs[pid], lambda: record(name, pairs[pid],
                        judge(root,pairs[pid],execution['data'],prediction['data']),mode='fresh',
                        parents=parents,source={'kind':'live_judge'},model=model),
                        parents=parents,model=model))
                    continue
                values.append(record(name,pairs[pid],payload,mode=actual_mode,
                                     parents=[execution['record_sha256'],prediction['record_sha256']],source=source_info,model=model))
            artifacts[name] = values
        checkpoint(name,artifacts[name])
    if journal is not None:
        manifest['reused_case_observations'] = journal.reused
    manifest.update(paid_api_calls_requested=bool(mode=='fresh' and stage in ('all','mutation','reasoning','judge')),
                    missing_judge_observations=missing_judge_observations(artifacts),
                    reasoning_null_to_error_normalizations=sum('normalization' in r['data'] for r in artifacts.get('reasoning',[])),
                    status='STAGES_COMPLETE')
    atomic_json(output/'manifest.json', manifest)
    # Validate the saved chain independently, including resumes, before calculating metrics.
    load_run(output)
    if stage in ('all','metrics'):
        manifest['metrics'] = calculate_metrics(root,output,artifacts,published)
        atomic_json(output/'manifest.json', manifest)
    manifest['status'] = 'COMPLETE'
    atomic_json(output/'manifest.json', manifest)
    lines = ['# MLIRBench pipeline reproduction','',
             'Default archived replay makes no model calls. Fresh stages are explicitly identified.','',
             '| Stage | Records | Mode | Work performed |','|---|---:|---|---|']
    descriptions = {'mutation':'Restore program pairs and mutation metadata' if all(r['mode']=='replay' for r in artifacts.get('mutation',[])) else 'Generate the selected mutation with the configured model',
                    'execution':'Restore historical observations' if all(r['mode']=='replay' for r in artifacts.get('execution',[])) else 'Compile and execute valid shared inputs locally',
                    'reasoning':'Restore saved model responses' if all(r['mode']=='replay' for r in artifacts.get('reasoning',[])) else 'Call the selected reasoning model',
                    'judge':'Recompute objective correctness and weighted totals from saved ratings' if all(r['mode']=='replay' for r in artifacts.get('judge',[])) else 'Run judge and compute weighted totals'}
    for name,values in artifacts.items():
        modes = ', '.join(sorted({r['mode'] for r in values}))
        description = descriptions[name]
        if name in manifest['reused_stages']:
            modes += ' (loaded checkpoint)'
            description = 'Reuse verified prior-stage output; no new stage call'
        lines.append(f'| {name} | {len(values)} | {modes} | {description} |')
    lines += ['', 'Each record retains source identities and selected parent hashes. Archive replay reconstructs the published record chain; it does not prove the unavailable historical runtime inputs. Published data and expected paper values are unchanged.','']
    if manifest['reasoning_null_to_error_normalizations']:
        lines += [f"Original failure-to-ERROR rule applied to {manifest['reasoning_null_to_error_normalizations']} archived null predictions. Raw null values and the normalization are retained explicitly in the reasoning records; published files are unchanged.",'']
    if 'judge' in artifacts:
        deltas = Counter(str(r['data'].get('score_delta',0)) for r in artifacts['judge'])
        lines += ['Saved subjective judge ratings are replayed, not regenerated. Total scores are calculated from components rather than copied from the paper. Archived totals are retained alongside the recomputed totals.', '',
                  'Recomputed-minus-archived per-record score deltas: '+json.dumps(dict(deltas),sort_keys=True)+'. Small differences can arise from archived component precision and historical floating-point rounding.', '',
                  f"Missing judge observations: {len(manifest['missing_judge_observations'])}; these are disclosed, never fabricated.", '']
    if journal is not None:
        lines += ['Durable fresh-case observations: cases/. Completed case reuse: '+json.dumps(journal.reused,sort_keys=True)+'.', '']
    if 'metrics' in manifest:
        lines += [f"Metric scope: {manifest['metrics']['scope']}.", '',
                  '[Computed metrics and paper comparison](metrics/REPORT.md)' if manifest['metrics']['scope']=='full_published_archive' else '[Computed sample/fresh metrics](metrics/summary.json)', '']
    lines += ['Fresh execution labels EQUIV mean agreement on the declared valid-input suite, not a proof for all inputs. Unsupported domains, timeouts and nonfinite results remain unscored.', '',
              'Inspect manifest.json and the stage JSONL files for inputs, source identities and parent hashes. These are reproducibility artifacts; this workflow writes no console-log files.', '']
    (output/'REPORT.md').write_text('\n'.join(lines))
    return {'output':str(output),'read_first':str(output/'REPORT.md'),
            'stages':{k:len(v) for k,v in artifacts.items()},'metrics':manifest.get('metrics'),
            'paid_api_calls_requested':manifest['paid_api_calls_requested'],
            'execution_labels':dict(Counter(str(r['data']['ground_truth_label']) for r in artifacts.get('execution',[])))}


def calculate_metrics(root, output, artifacts, published):
    judges = artifacts.get('judge',[])
    scored = [r for r in judges if r['data'].get('total_score') is not None]
    data = root/'data/processed/MLIRBench'
    all_replay = all(r['mode']=='replay' for values in artifacts.values() for r in values)
    expected_reasoning = {(r['program_id'],r['model_key']) for r in rows(data/VIEWS[1])} if all_replay else set()
    expected_judge = {(r['program_id'],r['evaluated_model']) for r in rows(data/VIEWS[2])} if all_replay else set()
    full = ({r['program_id'] for r in artifacts['mutation']} == set(published)
            and {(r['program_id'],r['model']) for r in artifacts.get('reasoning',[])} == expected_reasoning
            and {(r['program_id'],r['model']) for r in scored} == expected_judge
            and all(r['mode']=='replay' for values in artifacts.values() for r in values))
    if not full:
        folder = output/'metrics'; folder.mkdir()
        models = {}
        for model in sorted({r['model'] for r in judges}):
            records = [r['data'] for r in scored if r['model']==model]
            models[model] = {'scored':len(records),'mean_score':math.fsum(r['total_score'] for r in records)/len(records) if records else None}
        result = {'scope':'selected_or_fresh_records; not a reproduction of full paper tables','scored_records':len(scored),
                  'unscored_records':len(judges)-len(scored),'models':models}
        (folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    derived = output/'derived'; derived.mkdir()
    keep = ('program_id','dialect','level','seed','requirement_id','judge__outputs_diverge','pairs__status')
    master = [{k:r['data'][k] for k in keep} for r in artifacts['mutation']]
    evaluation = [r['data'] for r in scored]
    reasoning = [{'program_id':r['program_id'],'model_key':r['model']} for r in artifacts['reasoning']]
    for name,values in (('master',master),('evaluations',evaluation),('reasoning',reasoning)):
        with (derived/(name+'.jsonl')).open('x') as stream:
            for value in values: stream.write(json.dumps(value)+'\n')
    from .evaluation.reproduce import reproduce
    result = reproduce(root,output/'metrics',input_paths={k:derived/(k+'.jsonl') for k in ('master','evaluations','reasoning')})
    return {'scope':'full_published_archive',**result}
