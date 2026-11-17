"""Preserve private-repository visibility before any source-bearing image push."""
import json
import subprocess
from urllib.parse import quote


def gh(path):
    p=subprocess.run(['gh','api',path],capture_output=True,text=True)
    if p.returncode:
        if 'HTTP 404' in p.stderr:return None
        raise RuntimeError('Unable to verify registry visibility; no push permitted')
    return json.loads(p.stdout)


def main():
    import os
    repository=os.environ['GITHUB_REPOSITORY'];owner=repository.split('/')[0]
    repo=gh('repos/'+repository)
    if repo is None:raise RuntimeError('Repository metadata is unavailable')
    package=gh('users/'+owner+'/packages/container/'+quote('mlirbench-ae',safe=''))
    if repo['private'] and package is not None and package.get('visibility')!='private':
        raise RuntimeError('Private source cannot be pushed to an existing non-private image package')
    print(json.dumps({'repository_private':repo['private'],
                      'package_visibility':package.get('visibility') if package else 'new; GHCR defaults to private'}))


if __name__=='__main__':main()
