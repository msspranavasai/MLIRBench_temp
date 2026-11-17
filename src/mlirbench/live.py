"""Lazy adapters for original agent implementations. Imported only after live opt-in."""
import hashlib
import copy
import os
from pathlib import Path
import sys
from .observations import ObservationError, reject_credential_echo


def enable(root):
    folder = Path(root)/'optional/live_pipeline'
    if not folder.is_dir(): raise FileNotFoundError('Live agent source is not installed')
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))


def preflight(root, stages, models=None):
    """Resolve every requested model/credential/budget and construct SDKs without I/O.

    Only stages that will make fresh calls need credentials. A judge-only run
    resolves the evaluated reasoning model but does not require its API key.
    This validates local configuration, not remote model access or quota.
    """
    enable(root)
    from agents.api_config import model_id, request_limits
    from agents.llm_reasoning_agent.model_registry import get_spec, get_model_client
    stages = set(stages)
    specs = [get_spec(name) for name in (models or [])]
    config = {}
    if 'mutation' in stages:
        config['mutation'] = {'model_id': model_id('MLIRBENCH_MUTATION_MODEL', 'gpt-5-mini'),
                              'provider': 'OpenAI', 'api_key_env': 'OPENAI_API_KEY',
                              **request_limits('mutation', 8192)}
    if 'reasoning' in stages:
        if len(specs) != 1:
            raise ValueError('Choose exactly one reasoning model')
        spec = specs[0]
        config['reasoning'] = {'model_key': spec.key, 'model_id': spec.model_id,
                               'provider': spec.provider_label, 'api_key_env': spec.api_key_env,
                               'base_url': spec.base_url, **request_limits('reasoning', spec.max_tokens)}
    if 'judge' in stages:
        config['judge'] = {'model_id': model_id('OPENAI_MODEL', 'gpt-5.4-mini'),
                           'provider': 'OpenAI', 'api_key_env': 'OPENAI_API_KEY',
                           **request_limits('judge', 4096)}
    missing = sorted({c['api_key_env'] for c in config.values()
                      if not os.environ.get(c['api_key_env'], '').strip()})
    if missing:
        raise RuntimeError('Missing API credentials before any model calls: ' + ', '.join(missing))
    # Construct every required SDK now: optional dependency/configuration errors
    # must surface before an earlier agent can incur charges. No invoke/create.
    if 'mutation' in config:
        from agents.mutation_agent.llm import create_mutation_client
        client = create_mutation_client()
        client.close()
    if 'reasoning' in config:
        get_model_client(models[0])
    if 'judge' in config:
        from agents.judge_agent.node import _get_judge_llm
        _get_judge_llm()
    return {'stages': config, 'remote_access_verified': False,
            'note': 'Local configuration only; provider authentication, availability and quota are not probed.'}


def mutate(root, pair):
    enable(root)
    from agents.mutation_agent.node import mutation_agent
    result = mutation_agent(pair)['Agents']['Mutation_Agent']['context_results']
    if result.get('generation_status') not in ('VALIDATED','EXPECTED_INVALID'):
        raise ObservationError('Mutation generation failed; downstream agents were not called', result)
    reject_credential_echo(result)
    return {'program_id':pair['program_id'],'input_mlir':pair['input_mlir'],
            'mutated_mlir':result['mutated_mlir'],'dialect':pair['dialect'],'level':pair['level'],
            'seed':pair.get('seed',0),'requirement_id':result['requirement']['id'],
            'requirement_description':result['requirement']['description'],
            'mutation_observation':result,
            'model_id':os.environ.get('MLIRBENCH_MUTATION_MODEL','gpt-5-mini')}


def reason(root, pair, execution, model):
    enable(root)
    from agents.llm_reasoning_agent.node import reason_about_pair, _build_user_prompt
    from agents.llm_reasoning_agent.model_registry import get_spec
    inputs = execution.get('embedded_inputs') or {}
    result = reason_about_pair(pair['input_mlir'],pair['mutated_mlir'],pair['dialect'],
                               model,embedded_inputs=inputs,max_retries=0)
    if result.get('status') != 'SUCCESS':
        raise ObservationError('Reasoning call did not produce a valid response; no score fabricated: '+str(result.get('status')), result)
    reject_credential_echo(result)
    result['prompt_sha256'] = hashlib.sha256(_build_user_prompt(pair['input_mlir'],pair['mutated_mlir'],pair['dialect'],inputs).encode()).hexdigest()
    result['model_id'] = get_spec(model).model_id
    return result


def mutation_context(root, pair):
    """Restore the original rule/inspection context without inventing history."""
    enable(root)
    from .stage_artifacts import digest
    from agents.mutation_agent.selector import get_requirement_by_id
    dialect, level = pair.get('dialect'), pair.get('level')
    if dialect not in ('linalg','torch') or isinstance(level,bool) or level not in range(1,6):
        raise ValueError('Fresh judging needs the declared dialect and level 1..5')
    observation = pair.get('mutation_observation') or {}
    if not isinstance(observation, dict):
        raise ValueError('Invalid recorded mutation observation')
    recorded = observation.get('requirement')
    explicit = pair.get('mutation_rule')
    if recorded is not None and explicit is not None and recorded != explicit:
        raise ValueError('Conflicting recorded mutation rules')
    rule = recorded if recorded is not None else explicit
    rule_id = pair.get('requirement_id') or (rule.get('id') if isinstance(rule,dict) else None)
    if not isinstance(rule_id,str) or not rule_id:
        raise ValueError('Fresh judging needs an explicit mutation rule ID')
    # Validate the dialect/level/ID combination even when a full recorded rule
    # exists. A recorded rule is preserved, rather than replaced by today's text.
    catalog = get_requirement_by_id(dialect, level, rule_id)
    if rule is None:
        rule = catalog
        rule_source = 'packaged_rule_catalog; historical prompt identity unproven'
    else:
        rule_source = 'recorded_mutation_observation' if recorded is not None else 'explicit_recorded_rule'
    if (not isinstance(rule,dict) or rule.get('id') != rule_id
            or not isinstance(rule.get('description'),str) or not rule['description'].strip()
            or not isinstance(rule.get('rules'),list) or not rule['rules']
            or any(not isinstance(x,str) or not x.strip() for x in rule['rules'])):
        raise ValueError('Incomplete or mismatched mutation rule')
    metadata, metadata_source = {}, 'not_recorded'
    for owner, field, source in ((observation,'metadata','recorded_mutation_observation'),
                                  (pair,'pipeline_metadata','archived_pipeline_metadata'),
                                  (pair,'metadata','explicit_recorded_metadata')):
        if owner.get(field) is not None:
            if not isinstance(owner[field],dict):
                raise ValueError('Inspection metadata must be an object')
            metadata, metadata_source = owner[field], source
            break
    return {'mutation_id':rule_id, 'mutation_kind':rule['description'],
            'mutation_rule':copy.deepcopy(rule), 'metadata':copy.deepcopy(metadata),
            'mutation_context_provenance':{'rule_source':rule_source, 'rule_sha256':digest(rule),
                'metadata_source':metadata_source, 'metadata_sha256':digest(metadata)}}


def judge(root, pair, execution, prediction):
    if execution.get('ground_truth_label') is None:
        return {'program_id':pair['program_id'],'evaluated_model':prediction.get('model_key'),
                'status':'unscored_execution_unverified','total_score':None}
    enable(root)
    from agents.judge_agent.node import judge_agent, JudgeResponseError
    from .pipeline import score_components
    prediction = dict(prediction)
    prediction.setdefault('predicted_output_A',prediction.get('predicted_output_a'))
    prediction.setdefault('predicted_output_B',prediction.get('predicted_output_b'))
    state = {'judge_input':{
        **execution, 'program_id':pair['program_id'],'programA':pair['input_mlir'],'programB':pair['mutated_mlir'],
        'dialect':pair['dialect'], **mutation_context(root,pair),'prediction':prediction}}
    try:
        result = judge_agent(state)
    except (JudgeResponseError, ValueError):
        raise
    except Exception as exc:
        raise ObservationError('Judge API call failed; no score produced: '+str(exc),
                               {'status':'judge_call_failed','error':str(exc)}) from None
    reject_credential_echo(result)
    details = result.get('judge_details')
    if not details or details.get('judge_response_validation') != 'PASS':
        raise RuntimeError('Judge call failed or was skipped; no score fabricated')
    scores = details['llm_judge_scores']
    scored = score_components(prediction['predicted_label'],execution['ground_truth_label'],scores['result_accuracy'],scores['explanation_rating'])
    scored.update(program_id=pair['program_id'],evaluated_model=prediction.get('model_key'),
                  verdict=details['verdict'],judge_details=details,subjective_scores='fresh API judge observation')
    return scored
