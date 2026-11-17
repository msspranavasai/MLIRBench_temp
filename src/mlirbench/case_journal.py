"""Durable fresh-case observations, independent of stage checkpoint commits."""
import hashlib
import json
from pathlib import Path

from .observations import redact, reject_credential_echo
from .stage_artifacts import STAGES, atomic_json, digest, pair_digest, validate_record


def read_plan(folder, manifest):
    spec = manifest.get('case_plan')
    if not spec or spec.get('path') != 'case-plan.json':
        raise ValueError('Incomplete-call resume requires a case plan from a supported fresh run')
    raw = (Path(folder)/'case-plan.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']:
        raise ValueError('Modified fresh case plan')
    plan = json.loads(raw)
    if plan.get('schema') != 'mlirbench-case-plan-v1':
        raise ValueError('Unknown fresh case plan schema')
    return plan


def load_journal(folder, manifest, *, by_hash=None):
    """Validate successes and parent chains, including uncommitted stage output.

    Returned entries are resume evidence, not completed stage artifacts. Main
    load_run() deliberately does not expose a partial stage to metric consumers.
    """
    read_plan(folder, manifest)
    by_hash = dict(by_hash or {})
    entries = {}
    canonical_records = {(r['stage'],r['program_id'],r['model']):r for r in by_hash.values()}
    records = {}
    for stage in STAGES:
        for path in sorted((Path(folder)/'cases'/stage).glob('*.json')):
            entry = json.loads(path.read_text())
            copy = dict(entry); actual = copy.pop('entry_sha256', None)
            if digest(copy) != actual:
                raise ValueError('Modified case observation')
            request = entry['request']; key = digest(request)
            if (entry.get('schema') != 'mlirbench-case-v1' or request['stage'] != stage
                    or path.name != key+'.json' or request.get('mode') != 'fresh'):
                raise ValueError('Case observation identity differs')
            attempts = entry['attempts']
            if not attempts:
                raise ValueError('Case observation has no attempts')
            completed = False
            for index, attempt in enumerate(attempts):
                status = attempt['status']
                if status not in ('IN_FLIGHT','FAILED','INTERRUPTED','COMPLETE') or completed:
                    raise ValueError('Invalid case attempt history')
                if status != 'COMPLETE':
                    if 'record' in attempt:
                        raise ValueError('Unsuccessful case contains a completed record')
                    continue
                completed = True
                value = attempt['record']
                if (index != len(attempts)-1 or value['mode'] != 'fresh'
                        or value['parents'] != request['parents'] or value['model'] != request['model']):
                    raise ValueError('Case record does not match request')
                if stage != 'mutation' and (value['program_id'] != request['program_id']
                                             or value['pair_sha256'] != request['pair_sha256']):
                    raise ValueError('Case record belongs to a different input')
                if stage == 'mutation' and value['data'].get('comparison_program_id') != request['program_id']:
                    raise ValueError('Mutation case belongs to a different source')
                record_key = stage, value['program_id'], value['model']
                prior = records.get(record_key)
                if prior is not None:
                    raise ValueError('Duplicate completed case observation')
                canonical = canonical_records.get(record_key)
                if canonical is not None and canonical != value:
                    raise ValueError('Case observation differs from stage checkpoint')
                validate_record(value, stage, by_hash)
                records[record_key] = value
            entries[key] = entry
    return entries


class CaseJournal:
    def __init__(self, folder, plan, inherited=None):
        self.folder = Path(folder)
        self.entries = dict(inherited or {})
        self.reused = {stage: 0 for stage in STAGES}
        reject_credential_echo(plan)
        atomic_json(self.folder/'case-plan.json', plan)
        for entry in self.entries.values():
            self._save(entry)

    def plan_spec(self):
        return {'path':'case-plan.json',
                'sha256':hashlib.sha256((self.folder/'case-plan.json').read_bytes()).hexdigest()}

    def _save(self, entry):
        entry = dict(entry)
        entry.pop('entry_sha256', None)
        entry['entry_sha256'] = digest(entry)
        key = digest(entry['request'])
        directory = self.folder/'cases'/entry['request']['stage']
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory/(key+'.json'), entry)
        self.entries[key] = entry

    def call(self, stage, pair, function, *, parents=(), model=None):
        request = {'stage':stage, 'mode':'fresh', 'program_id':pair['program_id'],
                   'pair_sha256':pair_digest(pair) if 'mutated_mlir' in pair else None, 'input_sha256':digest(pair),
                   'parents':list(parents), 'model':model}
        reject_credential_echo(request)
        key = digest(request)
        entry = self.entries.get(key)
        if entry is not None and entry['attempts'][-1]['status'] == 'COMPLETE':
            self.reused[stage] += 1
            return entry['attempts'][-1]['record']
        entry = {'schema':'mlirbench-case-v1', 'request':request,
                 'attempts':list(entry['attempts']) if entry else []}
        entry['attempts'].append({'status':'IN_FLIGHT'})
        self._save(entry)
        try:
            value = function()
            reject_credential_echo(value)
        except BaseException as exc:
            entry['attempts'][-1].update(status='INTERRUPTED' if not isinstance(exc, Exception) else 'FAILED',
                                         failure_type=type(exc).__name__)
            if getattr(exc, 'observation', None) is not None:
                entry['attempts'][-1]['failure_observation'] = redact(exc.observation)
            self._save(entry)
            raise
        entry['attempts'][-1].update(status='COMPLETE', record=value)
        self._save(entry)
        return value
