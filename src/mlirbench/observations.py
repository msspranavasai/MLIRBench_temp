"""Sanitized diagnostics for new runs; frozen archived inputs are never rewritten."""
import json
import math
import os
import re
from urllib.parse import quote, quote_plus

REDACTED = '[REDACTED]'
SENSITIVE_FIELDS = {'authorization','proxy_authorization','proxy-authorization','api_key','apikey',
                    'api-key','access_token','refresh_token','password','secret','credentials',
                    'cookie','set-cookie','x-api-key','x-goog-api-key','request_headers','headers'}
AUTH_TEXT = re.compile(r'''(?ix)(\b(?:api[_-]?key|access[_-]?token|authorization|password|secret)["']?\s*[:=]\s*["']?)(?:Bearer\s+)?[^\s,"';}\]]+''')
BEARER = re.compile(r'''(?i)\bBearer\s+[^\s,"';}\]]+''')


def configured_secrets():
    values = set()
    for name,value in os.environ.items():
        if value and (name.endswith(('_API_KEY','_TOKEN','_PASSWORD','_SECRET')) or name == 'GEMINI_API_KEY'):
            values.update((value, quote(value,safe=''), quote_plus(value)))
    return sorted(values,key=len,reverse=True)


def contains_configured_secret(value):
    text = json.dumps(value,ensure_ascii=False,default=str) if not isinstance(value,str) else value
    return any(secret in text for secret in configured_secrets())


def redact(value):
    if isinstance(value,str):
        for secret in configured_secrets(): value=value.replace(secret,REDACTED)
        value=AUTH_TEXT.sub(lambda m:m.group(1)+REDACTED,value)
        return BEARER.sub('Bearer '+REDACTED,value)
    if isinstance(value,dict):
        return {str(k):REDACTED if str(k).lower() in SENSITIVE_FIELDS else redact(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [redact(x) for x in value]
    if isinstance(value,float) and not math.isfinite(value): return str(value)
    if value is None or isinstance(value,(bool,int,float)): return value
    return redact(str(value))


class ObservationError(RuntimeError):
    def __init__(self,message,observation):
        super().__init__(redact(message))
        self.observation=redact(observation)


def reject_credential_echo(value):
    if contains_configured_secret(value):
        raise ObservationError('Refusing to persist a configured credential in a fresh record',
                               {'status':'credential_echo_rejected'})
