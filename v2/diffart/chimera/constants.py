"""CGS constants. Reference: CHIMERA physics.rs, MIT.
Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
"""
RHO = 1.14e-3
C_SOUND = 35000.0
MU = 1.86e-4
NU = MU / RHO
RHO_C2 = RHO * C_SOUND**2
CM_H2O = 980.665
RE_CRIT = 1800.0
WALL_SOFT = (1.5, 1600., 3.e5)
WALL_TRACHEA = (.6, 1000., 1.e6)
WALL_NASAL = (1., 3000., 1.e6)
