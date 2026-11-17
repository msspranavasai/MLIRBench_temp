"""Explicit request budgets shared by preflight and the actual API clients."""
import math
import os


def integer(name, default, minimum, maximum):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an integer') from None
    if not minimum <= value <= maximum:
        raise ValueError(f'{name} must be between {minimum} and {maximum}')
    return value


def model_id(name, default):
    value = os.environ.get(name, default)
    if not value or any(c.isspace() for c in value) or not value.isprintable():
        raise ValueError(f'{name} must contain a nonempty model ID without whitespace')
    return value


def request_limits(stage, default_tokens):
    name = 'MLIRBENCH_API_TIMEOUT_SECONDS'
    try:
        timeout = float(os.environ.get(name, 180))
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be a finite number') from None
    if not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError(f'{name} must be finite and between 0 (exclusive) and 3600')
    return {
        'timeout_seconds': timeout,
        'max_output_tokens': integer(f'MLIRBENCH_{stage.upper()}_MAX_OUTPUT_TOKENS', default_tokens, 1, 131072),
        'sdk_retries': integer('MLIRBENCH_API_MAX_RETRIES', 0, 0, 5),
        'agent_attempts': integer('MLIRBENCH_MUTATION_MAX_ATTEMPTS', 3, 1, 10) if stage == 'mutation' else 1,
    }
