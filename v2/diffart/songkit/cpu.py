"""CPU execution of the unchanged CHIMERA acoustic graph, with exact replay state."""
from pathlib import Path
import importlib.util, json, math
import numpy as np
import torch
import torch.nn.functional as F
from diffart.data import HOP
from diffart.chimera.acoustics import bandlimit_noise
from diffart.chimera.coefficients import fixed_parameters, initial_state
from diffart.chimera.voice import HumanVoice
from diffart.chimera.kernel_graph import acoustic_graph
from diffart.chimera.constants import RHO_C2
from diffart.chimera import numba_graph
CACHE=Path('/tmp/chimera-songkit-cpu')
_NUMBA_FORWARD_BY_WIDTH={}
def compile_noisy_numba(net, width: int):
    """Compile the existing CPU Numba step, with the same generated step fed real noise."""
    if width in _NUMBA_FORWARD_BY_WIDTH:
        return _NUMBA_FORWARD_BY_WIDTH[width]
    cache_anchor = CACHE / '_runtime' / 'generated' / 'numba_graph.py'
    cache_anchor.parent.mkdir(parents=True, exist_ok=True)
    numba_graph.__file__ = str(cache_anchor)
    base_module = numba_graph.compile_forward(net, width)
    source_path = Path(base_module.__file__)
    source = source_path.read_text()
    old_signature = 'def forward(C,H,G,S0,hop):'
    if old_signature not in source:
        raise RuntimeError('Could not adapt the CHIMERA CPU Numba forward signature')
    source = source.replace(old_signature, 'def forward(C,H,G,S0,hop,Noise):', 1)
    if ',z,z,G[t,0])' not in source:
        raise RuntimeError('Could not route per-sample turbulence into the CHIMERA CPU step')
    source = source.replace(',z,z,G[t,0])', ',Noise[t],z,G[t,0])', 1)
    source = source.replace('    area=np.empty(samples)', '    area=np.empty(samples)\n    flow=np.empty(samples)')
    source = source.replace('        result[t]=y', '        result[t]=y\n        flow[t]=s1[' + str(net.glottis_edge) + ']/s2[' + str(net.glottis_edge) + ']')
    source = source.replace('return result,area', 'return result,area,flow')
    out_path = CACHE / 'numba_forward_with_noise.py'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source)
    spec = importlib.util.spec_from_file_location('ievan_numba_forward_with_noise', out_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    _NUMBA_FORWARD_BY_WIDTH[width] = module.forward
    return module.forward


def make_model(config: dict, vocab: dict, state: dict, scale_factor: float) -> HumanVoice:
    for key,expected in dict(mode='parametric',sample_rate=96000,sections=30,control_rate=1000,source_controls=True,wall_controls=True,source_shape_controls=True,conservative_geometry=True,tongue_modes=4,nasal_controls=True).items():
        if config.get(key)!=expected:raise ValueError('Unsupported inference architecture: '+key)
    model = HumanVoice(vocab=vocab)
    model.load_state_dict(state)
    model.eval().to(device='cpu', dtype=torch.float64 if config.get('precision') == 'torch.float64' else torch.float32)
    model.glottis.precise_phase = config.get('precise_phase', False)
    model.airway.correction_coefficient = config.get('correction_coefficient', 1.3067257008338073)
    with torch.no_grad():
        model.anatomy.log_scale.add_(math.log(scale_factor))
    return model


def cpu_render(model: HumanVoice, ctx: np.ndarray, pos: np.ndarray, f0: np.ndarray, seed: int):
    """Render on CPU with the model's same 96 kHz acoustic step and fixed band-limited noise."""
    phones = torch.as_tensor(ctx[None], dtype=torch.long, device='cpu')
    positions = torch.as_tensor(pos[None], dtype=model.pressure_raw.dtype, device='cpu')
    pitch = torch.as_tensor(f0[None], dtype=model.pressure_raw.dtype, device='cpu')
    with torch.inference_mode():
        single, controls, coefficients, glot, noise_gain, samples = model.prepare(phones, positions, pitch, seed=seed)
        if coefficients.shape[0] != 1:
            raise RuntimeError(f'Expected one CPU render batch, received {coefficients.shape[0]}')
        width = coefficients.shape[-1]
        d = len(model.airway.edge_a)
        white = torch.randn((1, samples, len(model.airway.noise_sources)), dtype=coefficients.dtype,
                            generator=torch.Generator(device='cpu').manual_seed(seed))
        noise = bandlimit_noise(white, model.sample_rate)
        if noise.shape[-1] != width:
            noise = F.pad(noise, (0, width - noise.shape[-1]))
        if isinstance(noise_gain, torch.Tensor) and noise_gain.ndim == 3 and noise_gain.shape[-1] > 1:
            noise_gain = F.pad(noise_gain, (0, width - noise_gain.shape[-1]), value=1.0)
        noise = noise.expand(1, -1, -1).contiguous() * noise_gain
        graph, states, coeffs, signals, headers, outputs, result = acoustic_graph(model.airway, width)
        del graph, coeffs, signals, outputs, result
        nstate = len(states)
        # The generated Numba graph contains double precision constants and
        # vectors. Keep every invocation array in float64 so Numba can carry
        # the solver state consistently through its sample loop.
        torch_noise = noise[0].contiguous().cpu().numpy().astype(np.float64)
        if torch_noise.shape[-1] != width:
            raise RuntimeError(f'Noise width {torch_noise.shape[-1]} does not match solver width {width}')
        c = coefficients[0].contiguous().cpu().numpy().astype(np.float64)
        h = fixed_parameters(model.airway, width, coefficients).contiguous().cpu().numpy().astype(np.float64)
        s0 = initial_state(model.airway, 1, width, coefficients).contiguous().cpu().numpy()[0].astype(np.float64)
        g = glot[0].contiguous().cpu().numpy().astype(np.float64)[:, None]
        forward = compile_noisy_numba(model.airway, width)
        wave96, fold_area, fold_flow = forward(c, h, g, s0, model.sample_rate // model.control_rate, torch_noise)
        if not np.isfinite(wave96).all():
            raise RuntimeError('CPU Numba acoustic output is non-finite')
        wave24 = model.decimator(torch.as_tensor(wave96[None], dtype=coefficients.dtype))[0]
        wave24 = (wave24 * model.microphone_log_gain.exp()).cpu().numpy().astype(np.float32)
        if len(wave24) != len(f0) * HOP:
            raise RuntimeError(f'Unexpected decimated audio length: {len(wave24)}')
        control_arrays = {}
        for key in ('states', 'outer', 'inner', 'areas', 'lengths', 'velum', 'minimum_areas', 'distances', 'source_offsets'):
            value = controls.get(key)
            if isinstance(value, torch.Tensor):
                control_arrays[key] = value[0].detach().cpu().numpy().copy()
        control_arrays['scale'] = np.asarray(float(controls['scale'].detach().cpu()), dtype=np.float32)
        nnode=len(model.airway.areas)
        control_arrays['solver_volumes'] = c[:,5,:nnode] * RHO_C2
        lengths64=c[:,15,:nnode]
        control_arrays['solver_areas'] = np.divide(control_arrays['solver_volumes'],lengths64,out=np.zeros_like(lengths64),where=lengths64>0)
        control_arrays['solver_lengths'] = lengths64.copy()
        control_arrays['glottal_area'] = fold_area.copy()
        control_arrays['glottal_flow'] = fold_flow.copy()
        control_arrays['glottal_rate'] = np.asarray(model.sample_rate)
        control_arrays['tube_indices_json'] = np.asarray(json.dumps(model.airway.tubes))
        body=(model.anatomy.body_logits.softmax(0) @ model.anatomy.bodies)
        control_arrays['body_json'] = np.asarray(json.dumps(dict(name='learned',scale=float(controls['scale']),pharynx=float(body[1]*model.anatomy.pharynx_scale.exp()),tongueRadius=float(body[2]),lipWidth=float(body[3]))))
        control_arrays['voiced'] = controls['voiced'][0].detach().cpu().numpy().astype(np.bool_)
        return wave24, control_arrays, dict(internal_samples=samples, solver_states=nstate,
                                             cpu_backend='CHIMERA Triton graph algebra via Numba CPU; 96 kHz; turbulence included')


