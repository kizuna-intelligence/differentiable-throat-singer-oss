"""Additional high-order checks near step resonances/antiresonances."""
import json
from pathlib import Path
import time

from .finite_volume import solve
from .multidomain import Piece,TubePINN
from .run_extended import complex_json,relative_field


def main():
    out=Path(__file__).resolve().parent/'results/extended'
    start=time.monotonic();rows=[]
    cases=[('axisymmetric',7500.,[Piece(0,.08,0,.01),Piece(.08,.17,0,.015)],True),
           ('asymmetric_slab',6000.,[Piece(0,.08,0,.02),Piece(.08,.17,0,.03)],False),
           ('asymmetric_slab',7500.,[Piece(0,.08,0,.02),Piece(.08,.17,0,.03)],False)]
    for name,f,pieces,axis in cases:
        ref=solve(pieces,f,spacing=.000125,axisymmetric=axis)
        order=[]
        for nx,ny in ((40,18),(48,24),(56,30)):
            m=TubePINN(pieces,axisymmetric=axis,nx=nx,ny=ny).train_variational(f)
            row=dict(nx=nx,ny=ny,transfer=complex_json(m.transfer()),field_relative_l2=relative_field(m,ref),
                transfer_relative_error=float(abs(m.transfer()/ref.transfer-1)))
            order.append(row);print(name,f,row,flush=True)
            m.save(out/f'{name}_{int(f)}_degree{nx}_{ny}.npz')
        rows.append(dict(geometry=name,frequency_hz=f,reference=complex_json(ref.transfer),spectral_orders=order))
    (out/'refined_steps.json').write_text(json.dumps(dict(elapsed_seconds=time.monotonic()-start,rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
