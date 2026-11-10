"""Summarize user-generated training/evaluation measurements as Markdown."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary',type=Path,nargs='?',default=Path(__file__).resolve().parents[1]/'artifacts/evaluation/summary.json',help='JSON from the evaluation workflow')
    parser.add_argument('--out',type=Path,default=Path('STATUS.md'))
    args=parser.parse_args()
    summary=json.loads(args.summary.read_text())
    lines=['# Training and evaluation status','','歌声データベース: 波音リツ（歌唱・提供: カノン）','']
    for key in ('training','db_reproduction','anatomy'):
        if key in summary:
            lines.extend(['## '+key,'','```json',json.dumps(summary[key],ensure_ascii=False,indent=2),'```',''])
    lines.extend(['These are saved measurements. Perceptual quality and anatomical identification are not established by these values.',''])
    args.out.write_text('\n'.join(lines))


if __name__=='__main__':
    main()
