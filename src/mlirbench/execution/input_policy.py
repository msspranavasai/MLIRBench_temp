"""Conservative, versioned input preflight. Unsupported programs never get a verdict.

The Torch evaluator checks domains only; its outputs are never used as ground truth.
Linalg currently supports static indexing and constant nonzero division denominators.
"""
import re
import ast
import struct
import numpy as np
from .execution import _parse_mlir_type

POLICY_VERSION = 'valid-domain-v5'
SSA = r'%[\w.$-]+'


class InputPolicyError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def unsupported(message):
    raise InputPolicyError('UNSUPPORTED_INPUT_PRECONDITIONS', message)


def invalid(message):
    raise InputPolicyError('INVALID_INPUT', message)


def torch_float_literal(text):
    text = text.strip()
    if re.fullmatch(r'0x[0-9A-Fa-f]{16}', text):
        return struct.unpack('>d', bytes.fromhex(text[2:]))[0]
    try:
        return float(text)
    except ValueError:
        unsupported('Unsupported Torch floating-point literal syntax')


def gather_bounds(source, names):
    """Resolve direct/unsqueezed input indices and static gather extents."""
    aliases = {name: i for i, name in enumerate(names)}
    constants = dict((m[0], int(m[1])) for m in re.findall(
        rf'({SSA})\s*=\s*torch.constant.int\s+(-?\d+)', source))
    bounds = {}
    for line in source.splitlines():
        if ' = torch.aten.unsqueeze ' in line or ' = torch.aten.squeeze.dim ' in line:
            lhs, rhs = line.split(' = ', 1)
            args = re.findall(SSA, rhs.split(' : ')[0])
            if args[0] in aliases:
                aliases[lhs.strip()] = aliases[args[0]]
        if ' = torch.aten.gather ' not in line:
            continue
        args = re.findall(SSA, line.split('torch.aten.gather', 1)[1].split(' : ')[0])
        types = re.findall(r'!torch.vtensor<\[([^\]]*)\],\s*(\w+)>', line)
        if len(args) != 4 or len(types) < 2 or args[1] not in constants or args[2] not in aliases:
            unsupported('gather requires static dimension and direct/unsqueezed input indices')
        try:
            shape = [int(x.strip()) for x in types[0][0].split(',') if x.strip()]
            dim = constants[args[1]]
            if dim < 0:
                dim += len(shape)
            if not 0 <= dim < len(shape):
                invalid('gather dimension out of range')
            bound = shape[dim]
        except ValueError:
            unsupported('dynamic gather shape')
        position = aliases[args[2]]
        bounds[position] = min(bounds.get(position, bound), bound)
    return bounds


def generate_input_case(a, b, specs, names_a, names_b, seed, dialect=None):
    """Search a bounded deterministic candidate suite, validating both programs.

    Sign and zero coverage is attempted where legal; positivity is a fallback
    when actual operations require it, not a global assumption about all inputs.
    """
    dialect = dialect or ('torch' if 'torch.' in a or 'torch.' in b else 'linalg')
    bounds = gather_bounds(a, names_a)
    for i, n in gather_bounds(b, names_b).items():
        bounds[i] = min(bounds.get(i, n), n)
    patterns = ('signed_normal', 'positive_uniform', 'negative_uniform', 'zeros', 'ones', 'small_signed', 'positive_large')
    first = patterns[seed % 4]
    order = (first,) + tuple(p for p in patterns if p != first)
    attempts = []
    for attempt in range(21):
        pattern = order[attempt % len(order)]
        rng = np.random.default_rng(np.random.SeedSequence([seed, attempt]))
        values = []
        for i, spec in enumerate(specs):
            dtype, shape = np.dtype(spec['dtype']), tuple(spec['shape'])
            if i in bounds:
                if dtype.kind not in 'iu' or bounds[i] <= 0:
                    invalid('gather index type or extent invalid')
                v = rng.integers(0, bounds[i], size=shape, dtype=dtype)
            elif pattern in ('zeros', 'ones'):
                v = np.full(shape, int(pattern == 'ones'), dtype=dtype)
            elif dtype.kind == 'b':
                v = rng.integers(0, 2, size=shape).astype(dtype)
            elif dtype.kind == 'f':
                if pattern == 'signed_normal': v = rng.normal(0, 1, size=shape)
                elif pattern == 'positive_uniform': v = rng.uniform(.25, 2, size=shape)
                elif pattern == 'negative_uniform': v = rng.uniform(-2, -.25, size=shape)
                elif pattern == 'small_signed': v = rng.uniform(-.25, .25, size=shape)
                else: v = rng.uniform(128, 256, size=shape)
                v = v.astype(dtype)
            else:
                low, high = ((1, 5) if pattern in ('positive_uniform', 'positive_large')
                             else (-4, 0) if pattern == 'negative_uniform' else (-4, 5))
                if dtype.kind == 'u': low, high = max(0, low), max(1, high)
                v = rng.integers(low, high, size=shape, dtype=dtype)
            values.append(np.asarray(v))
        try:
            check_inputs(a, names_a, values, dialect)
            check_inputs(b, names_b, values, dialect)
        except InputPolicyError as exc:
            if exc.status != 'INVALID_INPUT':
                raise
            attempts.append({'attempt': attempt, 'pattern': pattern, 'reason': str(exc)})
            continue
        return values, {'strategy': 'deterministic_domain_checked_candidates',
                        'seed': seed, 'attempt': attempt, 'pattern': pattern,
                        'rejected_candidates': attempts}
    raise InputPolicyError('INPUT_GENERATION_FAILED',
                           'No valid shared candidate in the bounded suite; last failure: ' + attempts[-1]['reason'])


def generate_inputs(a, b, specs, names_a, names_b, seed):
    return generate_input_case(a, b, specs, names_a, names_b, seed)[0]


def _torch_preflight(source, inputs):
    if len(re.findall(r'func.func\s+', source)) != 1:
        unsupported('Torch helper functions require a separate domain analysis')
    env = dict(inputs)
    def value(token):
        token = token.strip()
        if token not in env:
            unsupported(f'unresolved Torch operand {token}')
        return env[token]
    parsed_operations = 0
    for line in source.splitlines():
        m = re.match(r'\s*(%[^=]+?)\s*=\s*(torch\.[\w.]+)\s*(.*)', line)
        if not m:
            continue
        parsed_operations += 1
        lhs, op, rest = m.groups()
        args = [value(t) for t in re.findall(SSA, rest.split(' : ')[0])]
        try:
            with np.errstate(all='raise', under='ignore'):
                if op == 'torch.vtensor.literal':
                    literal = re.search(r'dense<(.*?)>\s*:\s*(tensor<[^>]+>)', rest)
                    if not literal: unsupported('unsupported Torch literal syntax')
                    spec = _parse_mlir_type(literal[2])
                    if spec is None: unsupported('unsupported Torch literal type')
                    try: data = ast.literal_eval(literal[1].replace('true', 'True').replace('false', 'False'))
                    except (SyntaxError, ValueError): unsupported('unsupported Torch literal value')
                    out = np.asarray(data, dtype=spec['dtype'])
                    shape = tuple(spec['shape'])
                    out = np.full(shape, out, dtype=spec['dtype']) if out.ndim == 0 else out.reshape(shape)
                elif op == 'torch.constant.device':
                    if rest.strip() != '"cpu"': unsupported('only CPU device constants supported')
                    out = 'cpu'
                elif op == 'torch.constant.int': out = int(rest)
                elif op == 'torch.constant.float': out = torch_float_literal(rest)
                elif op == 'torch.constant.bool':
                    if rest.strip() not in ('true', 'false'): unsupported('Unsupported Torch boolean literal syntax')
                    out = rest.strip() == 'true'
                elif op == 'torch.constant.none': out = None
                elif op == 'torch.prim.ListConstruct': out = args
                elif op in ('torch.aten.add.Tensor', 'torch.aten.add.Scalar'): out = args[0] + args[1] * args[2]
                elif op in ('torch.aten.sub.Tensor', 'torch.aten.sub.Scalar'): out = args[0] - args[1] * args[2]
                elif op == 'torch.aten.rsub.Scalar': out = args[1] - args[0] * args[2]
                elif op in ('torch.aten.mul.Tensor', 'torch.aten.mul.Scalar'): out = args[0] * args[1]
                elif op in ('torch.aten.div.Tensor', 'torch.aten.div.Scalar'):
                    if np.any(np.asarray(args[1]) == 0): invalid('zero divisor')
                    out = args[0] / args[1]
                elif op == 'torch.aten.log':
                    if np.any(np.asarray(args[0]) <= 0): invalid('nonpositive log argument')
                    out = np.log(args[0])
                elif op == 'torch.aten.sqrt':
                    if np.any(np.asarray(args[0]) < 0): invalid('negative sqrt argument')
                    out = np.sqrt(args[0])
                elif op == 'torch.aten.exp': out = np.exp(args[0])
                elif op == 'torch.aten.abs': out = np.abs(args[0])
                elif op == 'torch.aten.relu': out = np.maximum(args[0], 0)
                elif op == 'torch.aten.neg': out = -args[0]
                elif op == 'torch.aten.sum': out = np.sum(args[0])
                elif op == 'torch.aten.sum.dim_IntList': out = np.sum(args[0], axis=tuple(args[1]), keepdims=args[2])
                elif op in ('torch.aten.max.dim', 'torch.aten.min.dim'):
                    f, g = (np.max, np.argmax) if '.max.' in op else (np.min, np.argmin)
                    idx = g(args[0], axis=args[1])
                    if args[2]: idx = np.expand_dims(idx, args[1])
                    out = (f(args[0], axis=args[1], keepdims=args[2]), idx.astype(np.int64))
                elif op == 'torch.aten.max': out = np.max(args[0])
                elif op == 'torch.aten.min': out = np.min(args[0])
                elif op == 'torch.aten.all': out = np.all(args[0])
                elif op == 'torch.aten.all.dim': out = np.all(args[0], axis=args[1], keepdims=args[2])
                elif re.fullmatch(r'torch.aten.(eq|ne|gt|ge|lt|le).(Tensor|Scalar)', op):
                    out = {'eq': np.equal, 'ne': np.not_equal, 'gt': np.greater, 'ge': np.greater_equal, 'lt': np.less, 'le': np.less_equal}[op.split('.')[2]](args[0], args[1])
                elif op == 'torch.aten.to.dtype':
                    dtypes = {3: np.int32, 4: np.int64, 5: np.float16, 6: np.float32, 7: np.float64, 11: np.bool_}
                    if args[1] not in dtypes: unsupported('unsupported Torch cast')
                    out = np.asarray(args[0]).astype(dtypes[args[1]])
                elif op == 'torch.aten.unsqueeze': out = np.expand_dims(args[0], args[1])
                elif op == 'torch.aten.squeeze.dim': out = np.squeeze(args[0], axis=args[1])
                elif op == 'torch.aten.gather':
                    data, dim, indices = args[:3]
                    if indices.dtype.kind not in 'iu' or indices.ndim != data.ndim: invalid('gather index rank/dtype')
                    if np.any(indices < 0) or np.any(indices >= data.shape[dim]): invalid('gather indices out of bounds')
                    if any(indices.shape[d] > data.shape[d] for d in range(data.ndim) if d != dim % data.ndim): invalid('gather index shape')
                    out = np.take_along_axis(data, indices, axis=dim)
                elif op in ('torch.aten.mm', 'torch.aten.matmul'): out = np.matmul(args[0], args[1])
                elif op == 'torch.aten.where.self': out = np.where(*args)
                elif op == 'torch.aten.clamp': out = np.clip(*args)
                elif op == 'torch.aten.permute': out = np.transpose(args[0], args[1])
                elif op == 'torch.aten.view': out = np.reshape(args[0], args[1])
                elif op == 'torch.aten.cat': out = np.concatenate(args[0], axis=args[1])
                elif op == 'torch.aten.arange.start_step':
                    if args[2] == 0 or abs((args[1] - args[0]) / args[2]) > 1000000: invalid('invalid or excessive arange')
                    out = np.arange(*args[:3], dtype=np.int64)
                elif op == 'torch.aten.broadcast_to': out = np.broadcast_to(args[0], args[1])
                elif op == 'torch.aten.flip': out = np.flip(args[0], axis=tuple(args[1]))
                elif op == 'torch.aten.transpose.int': out = np.swapaxes(*args[:3])
                elif op == 'torch.aten.cumsum': out = np.cumsum(args[0], axis=args[1])
                elif op == 'torch.aten.pow.Tensor_Scalar':
                    if np.any(args[0] < 0) and args[1] != int(args[1]): invalid('fractional power of negative input')
                    if args[1] < 0 and np.any(args[0] == 0): invalid('negative power of zero')
                    out = np.power(args[0], args[1])
                elif op == 'torch.aten.slice.Tensor':
                    if args[4] <= 0: invalid('slice step must be positive')
                    sl = [slice(None)] * args[0].ndim; sl[args[1]] = slice(*args[2:5]); out = args[0][tuple(sl)]
                else: unsupported(f'no Torch preflight for {op}')
        except FloatingPointError as exc:
            invalid(f'{op}: {exc}')
        except (IndexError, ValueError, TypeError) as exc:
            if isinstance(exc, InputPolicyError): raise
            unsupported(f'Cannot validate source operation {op}: {exc}')
        outputs = out if isinstance(out, tuple) else (out,)
        names = [n.strip() for n in lhs.split(',')]
        if len(names) != len(outputs): unsupported(f'result mapping for {op}')
        env.update(zip(names, outputs))
    if parsed_operations != len(re.findall(r'=\s*"?torch\.[\w.]+', source)):
        unsupported('Unparsed Torch operations; use one supported operation per line')


def _linalg_preflight(source):
    # No attempt to interpret arbitrary affine maps, helper calls or dynamic indexing.
    if len(re.findall(r'func.func\s+', source)) != 1:
        unsupported('Linalg helper functions require a separate domain analysis')
    if re.search(r'\b(math\.(log|log2|log10|sqrt|rsqrt|fpowi|powf)|arith\.(div[su]i|ceildiv[su]i|floordivsi|rem[su]i|fptosi|fptoui)|linalg\.gather)\b', source):
        unsupported('Linalg log/sqrt/power/integer-domain operation requires analysis')
    if re.search(r'tensor\.(extract|insert)\s', source) or re.search(r'tensor\.(extract_slice|insert_slice)[^\n]*\[[^\]\n]*%', source):
        unsupported('dynamic or scalar tensor indexing requires analysis')
    if re.search(r'\b(memref\.|scf\.|cf\.|vector\.)', source):
        unsupported('unanalysed memory/control-flow operation')
    for mapping in re.findall(r'affine_map<\([^>]*?->\s*\((.*?)\)>', source):
        if any(not re.fullmatch(r'\s*(d\d+|0)\s*', part) for part in mapping.split(',') if part.strip()):
            unsupported('Non-projection affine indexing map requires bounds analysis')
    constants = set()
    for m in re.finditer(rf'({SSA})\s*=\s*arith.constant\s+(?:dense<)?([-+\d.eE]+)([^\n]*)', source):
        declared = re.search(r':\s*(?:tensor<)?(?:\d+x)*(f16|f32|f64)>?', m[3])
        if not declared:
            continue
        try:
            with np.errstate(all='ignore'):
                value = np.asarray(float(m[2]), dtype={'f16': 'float16', 'f32': 'float32', 'f64': 'float64'}[declared[1]])
            if np.isfinite(value) and value != 0: constants.add(m[1])
        except ValueError: pass
    # Bind scalar region arguments to constant tensors, scoped per generic region.
    checked = 0
    for region in re.finditer(r'linalg.generic\b(.*?)\^bb0\((.*?)\):(.*?)linalg.yield', source, re.S):
        head, block, body = region.groups()
        ins = re.search(r'ins\((.*?)\s*:', head, re.S)
        outs = re.search(r'outs\((.*?)\s*:', head, re.S)
        operands = re.findall(SSA, ins[1] if ins else '') + re.findall(SSA, outs[1] if outs else '')
        params = re.findall(rf'({SSA})\s*:', block)
        # Output/reduction block arguments can change across iterations even if
        # initialized from a nonzero constant. Only immutable inputs are trusted.
        ins_operands = re.findall(SSA, ins[1] if ins else '')
        local = {p for p, v in zip(params, ins_operands) if v in constants}
        for m in re.finditer(rf'arith.divf\s+{SSA}\s*,\s*({SSA})', body):
            checked += 1
            if m[1] not in local and not (m[1] in constants and m[1] not in params):
                unsupported('Linalg runtime divisor requires analysis')
    if checked != len(re.findall(r'\barith.divf\b', source)):
        unsupported('unresolved Linalg division region')


def check_inputs(source, names, values, dialect):
    if any(not np.all(np.isfinite(v)) for v in values): invalid('nonfinite input')
    if dialect == 'torch': _torch_preflight(source, dict(zip(names, values)))
    elif dialect == 'linalg': _linalg_preflight(source)
    else: unsupported(f'unknown dialect {dialect}')
