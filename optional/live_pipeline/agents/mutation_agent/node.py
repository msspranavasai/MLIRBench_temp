from mlirbench.observations import redact, reject_credential_echo
import re
from agents.mutation_agent.selector import pick_requirement, get_requirement_by_id
from agents.mutation_agent.prompts.prompt_builder import build_user_prompt
from agents.mutation_agent.llm import generate_mutation
# from services.mlir.validator_client import validate_mutation, validate_torch_mutation
from services.mlir.validator_client import (
    ValidatorInfrastructureError,
    validate_mutation,
    validate_torch_mutation,
)
# from services.dataset.logger import save_run

from agents.mutation_agent.inspector.linalg.level1 import inspect_level1
from agents.mutation_agent.inspector.linalg.level2 import inspect_level2
from agents.mutation_agent.inspector.linalg.level3 import inspect_level3
from agents.mutation_agent.inspector.linalg.level4 import inspect_level4
from agents.mutation_agent.inspector.linalg.level5 import inspect_level5

from agents.mutation_agent.inspector.torch.level1 import torch_inspect_level1
from agents.mutation_agent.inspector.torch.level2 import torch_inspect_level2
from agents.mutation_agent.inspector.torch.level3 import torch_inspect_level3
from agents.mutation_agent.inspector.torch.level4 import torch_inspect_level4
from agents.mutation_agent.inspector.torch.level5 import torch_inspect_level5


LINALG_INSPECTORS = {
    1: inspect_level1,
    2: inspect_level2,
    3: inspect_level3,
    4: inspect_level4,
    5: inspect_level5
}

TORCH_INSPECTORS = {
    1: torch_inspect_level1,
    2: torch_inspect_level2,
    3: torch_inspect_level3,
    4: torch_inspect_level4,
    5: torch_inspect_level5
}


from agents.api_config import request_limits


def complete_module(candidate):
    """Envelope check only; the compiler still decides validity, not this check."""
    if not isinstance(candidate, str): return False
    # Ignore quoted strings and comments when balancing the module envelope.
    text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', '', candidate, flags=re.S).strip()
    if not re.match(r'^(?:builtin\.)?module\b', text) or not re.search(r'\bfunc\.func\s+@', text) or not text.endswith('}'):
        return False
    depth = 0
    for c in text:
        if c == '{': depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0: return False
    return depth == 0 and '{' in text


def mutation_agent(payload: dict):
    input_mlir, dialect, level = payload['input_mlir'], payload['dialect'], payload['level']
    reject_credential_echo(payload)
    if dialect not in ('linalg', 'torch') or type(level) is not int or level not in range(1, 6):
        raise ValueError('Mutation needs dialect linalg|torch and level 1..5')
    metadata = (LINALG_INSPECTORS if dialect == 'linalg' else TORCH_INSPECTORS)[level](input_mlir)
    requirement = (get_requirement_by_id(dialect, level, payload['requirement_id'])
                   if payload.get('requirement_id') else pick_requirement(input_mlir, dialect, level))
    validator = validate_mutation if dialect == 'linalg' else validate_torch_mutation
    # Level 5 deliberately generates invalid programs. Require a valid original
    # and an explicit catalog rule, so generation failures cannot masquerade as it.
    if level == 5:
        if not payload.get('requirement_id'):
            raise ValueError('Fresh Level 5 requires an explicit catalog rule ID')
        source_validation = validator(input_mlir, input_mlir)
        if not source_validation.get('valid'):
            raise ValueError('Fresh Level 5 requires a compiler-valid original program')
    prompt = build_user_prompt(input_mlir, requirement, metadata)
    mutated_mlir, status, history = 'GENERATION_FAILED', 'FAILED', []
    for attempt in range(1, request_limits('mutation', 8192)['agent_attempts'] + 1):
        try:
            candidate = generate_mutation(prompt)
            reject_credential_echo(candidate)
            validation = validator(input_mlir, candidate)
            expected_invalid = (level == 5 and candidate != input_mlir and complete_module(candidate)
                                and validation.get('valid') is False
                                and type(validation.get('returncode')) is int
                                and validation['returncode'] > 0
                                and (validation.get('diagnostic_kind') == 'verification_error'
                                     or (requirement['id'].endswith('_R4')
                                         and validation.get('diagnostic_kind') == 'ssa_error')))
            accepted = expected_invalid if level == 5 else validation.get('valid') is True
            history.append(redact({'attempt':attempt, 'candidate':candidate, 'validation':validation,
                                   'accepted':accepted}))
            if accepted:
                mutated_mlir, status = candidate, 'EXPECTED_INVALID' if expected_invalid else 'VALIDATED'
                break
        except ValidatorInfrastructureError:
            raise  # More paid generation cannot repair missing compiler infrastructure.
        except Exception as exc:
            history.append(redact({'attempt':attempt, 'candidate':None,
                                   'validation':{'valid':False, 'error':str(exc)}}))
    observation = {'level':level, 'dialect':dialect, 'requirement':requirement,
        'generation_status':status, 'validator_mode':'ordinary_compiler_verification',
        # Retain the legacy field name for readers; the actual validator mode is explicit.
        'mlir-opt-verify-diagnostics':{'attempts':len(history),
            'final_validation':history[-1]['validation'] if history else {}, 'history':history},
        'input_mlir':input_mlir, 'mutated_mlir':mutated_mlir, 'metadata':metadata}
    return {'Agents':{'Mutation_Agent':{'context_results':observation,
        'blind_results':{'input_mlir':input_mlir, 'mutated_mlir':mutated_mlir},
        'toExecutionAgent':{'input_mlir':input_mlir, 'mutated_mlir':mutated_mlir, 'dialect':dialect}}}}
