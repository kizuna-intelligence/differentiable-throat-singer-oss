"""HumanVoice: 38 Japanese gestures -> moving 2D anatomy -> physical waveform.

The training and inference interface accepts three-phone context,
phone position and given F0; learned audio residuals are deliberately absent.
Physical constants and topology: CHIMERA voice/human_airway.rs, MIT,
Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .anatomy import HumanAnatomy
from .articulators import ArticulatorDynamics, NEUTRAL
from .gestures import GesturalScore
from .glottis import ParametricGlottis
from .human_airway import build_human_airway
from .decimator import HalfBandDecimator
from .coefficients import pack_coefficients
from .constants import CM_H2O

class HumanVoice(nn.Module):

    def __init__(self, vocab):
        super().__init__()
        self.vocab = dict(vocab)
        self.mode = 'parametric'
        self.sample_rate = 96000
        self.sr = 24000
        self.hop = 240
        self.control_rate = 1000
        self.use_source_controls = True
        self.source_shape_controls = True
        self.given_voicing = False
        self.lead_frames = int(0)
        self.use_wall_controls = True
        self.use_branch_controls = False
        self.use_nasal_controls = True
        self.nasal_calibration = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.wall_calibration = nn.Parameter(torch.zeros(3, dtype=torch.float64))
        self.source_net = nn.Sequential(nn.Linear(52, 32, dtype=torch.float64), nn.Tanh(), nn.Linear(32, 6, dtype=torch.float64))
        nn.init.zeros_(self.source_net[-1].weight)
        nn.init.zeros_(self.source_net[-1].bias)
        self.score = GesturalScore(vocab)
        self.dynamics = ArticulatorDynamics(100)
        self.tongue_modes = 4
        self.tongue_targets = nn.Parameter(torch.zeros(len(vocab), 4, dtype=torch.float64))
        self.tongue_net = nn.Sequential(nn.Linear(51, 32, dtype=torch.float64), nn.Tanh(), nn.Linear(32, 4, dtype=torch.float64))
        nn.init.zeros_(self.tongue_net[-1].weight)
        nn.init.zeros_(self.tongue_net[-1].bias)
        self.tongue_dynamics = ArticulatorDynamics(100, speeds=(10,) * 4, bounds=((-1.0, 1.0),) * 4, neutral=(0.0,) * 4)
        self.anatomy = HumanAnatomy(sections=30 or 30, conservative=True, shape_controls=False)
        with torch.no_grad():
            geo = self.anatomy(torch.tensor(NEUTRAL, dtype=torch.float64)[None, None])
        self.airway = build_human_airway(geo['areas'][0, 0].numpy(), float(geo['lengths'][0, 0].sum()), float(geo['scale']), sample_rate=96000)
        self.airway.frequency_dependent_loss = True
        self.airway.consistent_loss_geometry = True
        self.airway.interpolate_coefficients = True
        self.airway.correction_coefficient = 1.3067257008338073
        self.airway.nasal_correction_coefficient = 0.0
        self.airway.glottis_mode = 'parametric'
        self.glottis = ParametricGlottis(96000)
        self.decimator = HalfBandDecimator(96000 // self.sr)
        self.pressure_raw = nn.Parameter(torch.tensor(math.log(5 / 8), dtype=torch.float64))
        self.noise_raw = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.microphone_log_gain = nn.Parameter(torch.tensor(math.log(1e-08), dtype=torch.float64))
        self.register_buffer('reference_scale', geo['scale'].detach())
        self.register_buffer('oral_indices', torch.tensor(self.airway.tubes['pharynx'] + self.airway.tubes['oral']))
        self.register_buffer('pause_id', torch.tensor(vocab['pau']))

    def controls(self, phones, pos):
        targets, strength, voiced = self.score(phones, pos)
        advanced = self.dynamics(targets, strength)
        states = torch.cat((self.dynamics.neutral[None, None].expand(targets.shape[0], 1, -1), advanced[:, :-1]), 1)
        extra = {}
        features = torch.cat((self.score.embedding(phones).flatten(-2), pos[..., None], torch.sin(math.pi * pos)[..., None], torch.cos(math.pi * pos)[..., None]), -1)
        goal = torch.tanh(self.tongue_targets[phones[..., 1]] + self.tongue_net(features))
        evolved = self.tongue_dynamics(goal)
        tongue = torch.cat((torch.zeros_like(evolved[:, :1]), evolved[:, :-1]), 1)
        geometry = self.anatomy(states, tongue_states=tongue)
        extra = dict(tongue_targets=goal, tongue_states=tongue)
        geometry['nasal_area'] = geometry['nasal_area'] * torch.exp(math.log(2) * self.nasal_calibration[0].tanh())
        geometry['nasal_length_scale'] = torch.exp(0.3 * self.nasal_calibration[1].tanh())
        return dict(targets=targets, strength=strength, states=states, voiced=voiced, **extra, **geometry)

    def acoustic_coefficients(self, controls, phones):
        geo = controls
        net = self.airway
        b, f, noral = geo['areas'].shape
        ratio = geo['scale'] / self.reference_scale
        areas = (net.areas * ratio.square())[None, None].expand(b, f, -1).clone()
        lengths = (net.dx * ratio)[None, None].expand(b, f, -1).clone()
        areas[:, :, self.oral_indices] = geo['areas']
        lengths[:, :, self.oral_indices] = geo['lengths']
        ix = net.tubes['nasal']
        areas[:, :, ix] = areas[:, :, ix] * torch.exp(math.log(2) * self.nasal_calibration[0].tanh())
        lengths[:, :, ix] = lengths[:, :, ix] * torch.exp(0.3 * self.nasal_calibration[1].tanh())
        count = f * self.control_rate // 100

        def up(value):
            return F.interpolate(value.transpose(1, 2), size=count, mode='linear', align_corners=False).transpose(1, 2)
        minimum = None
        if 'minimum_areas' in geo:
            minimum = areas.clone()
            minimum[:, :, self.oral_indices] = geo['minimum_areas']
            minimum = up(minimum)
        areas = up(areas)
        lengths = up(lengths)
        ports = net.port_area[None, None].expand(b, count, -1).clone()
        ports[:, :, net.velum_edge] = up(geo['velum'][..., None])[..., 0]
        c = pack_coefficients(net, areas, lengths, ports, minimum_areas=minimum)
        scales = torch.exp(math.log(4) * self.wall_calibration.tanh())
        c = torch.cat((c[:, :, :8], c[:, :, 8:11] * scales[None, None, :, None], c[:, :, 11:]), 2)
        active = (phones[..., 1] != self.pause_id).to(areas.dtype)
        active = up(active[..., None])[..., 0]
        offset = geo.get('source_offsets')
        pressure = (3.0 + 13.0 * self.pressure_raw.sigmoid()) * CM_H2O
        if offset is not None:
            raw = self.pressure_raw + torch.zeros_like(offset[..., 0])
            delta = 13 * CM_H2O * ((raw + offset[..., 0]).sigmoid() - raw.sigmoid())
            pressure = pressure + up(delta[..., None])[..., 0]
        fixedp = c[:, :, 13].clone()
        lung = net.names.index('lungs')
        fixedp[:, :, lung] = active * pressure
        c = c.clone()
        c[:, :, 13] = fixedp
        return c

    def prepare(self, phones, pos, f0, seed=0, rd_override=None, return_controls=False, phase_offset=0.0, return_fold_area=False):
        single = f0.ndim == 1
        if single:
            phones, pos, f0 = (phones[None], pos[None], f0[None])
        dtype = self.pressure_raw.dtype
        pos = pos.to(dtype)
        f0 = f0.to(dtype)
        p = self.controls(phones, pos)
        features = torch.cat((self.score.embedding(phones).flatten(-2), pos[..., None], torch.sin(math.pi * pos)[..., None], torch.cos(math.pi * pos)[..., None], torch.log2(f0.clamp_min(30) / 220)[..., None]), -1)
        p['source_offsets'] = 2 * self.source_net(features).tanh()
        c = self.acoustic_coefficients(p, phones)
        samples = f0.shape[1] * self.sample_rate // 100
        freq = F.interpolate(f0[:, None], size=samples, mode='linear', align_corners=False)[:, 0]
        if rd_override is not None:
            raise ValueError('Rd is an LF-source parameter; this source requires its own measured shape controls')
        offsets = p.get('source_offsets')
        if offsets is None:
            ga = self.glottis(freq, seed=seed, phase_offset=phase_offset)
        else:
            source = F.interpolate(offsets.transpose(1, 2), size=samples, mode='linear', align_corners=False).transpose(1, 2)
            extra = dict(skew_offset=source[..., 4], sharpness_offset=source[..., 5])
            ga = self.glottis(freq, seed=seed, phase_offset=phase_offset, area_offset=source[..., 1], oq_offset=source[..., 2], **extra)
        voiced = F.interpolate(p['voiced'][:, None].to(dtype), size=samples, mode='nearest')[:, 0]
        ga = voiced * ga + (1 - voiced) * (0.12 + F.softplus(self.glottis.chink_raw))
        noise_gain = torch.exp(2 * self.noise_raw.tanh())
        mod = F.interpolate(p['source_offsets'][..., 3][:, None], size=samples, mode='linear', align_corners=False)[:, 0]
        noise_gain = noise_gain * mod.exp()[..., None]
        return (single, p, c, ga, noise_gain, samples)

    def forward(self, phones, pos, f0, seed=0, return_controls=False):
        from .training_backend import render
        single, controls, c, ga, gain, samples = self.prepare(phones, pos, f0, seed=seed)
        wave = render(self.airway, c, ga, gain, seed, self.sample_rate // self.control_rate)
        wave = self.decimator(wave) * self.microphone_log_gain.exp()
        if single:
            wave = wave[0]
            controls = {k: (v[0] if v.ndim and v.shape[0] == 1 else v) for k,v in controls.items()}
        return (wave, controls) if return_controls else wave

    def regularization(self, controls):
        areas = controls['areas']
        spatial = (areas[..., 1:] - areas[..., :-1]).square().mean()
        temporal = ((areas[..., 1:, :] - areas[..., :-1, :]).square().mean()
                    if areas.shape[-2] > 1 else areas.sum() * 0)
        return 1e-3 * (spatial + temporal) + 1e-4 * self.wall_calibration.square().mean()
