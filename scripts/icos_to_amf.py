"""Write an ICOS station's half-hourly data in the AmeriFlux BASE layout, so the AmeriFlux scripts (field_retention.py,
prepare_site_amf.py, find_window.py, compare_wells.py, ...) run on it unchanged:
  <out>_BASE_HH.csv: two comment lines, then TIMESTAMP_START, TIMESTAMP_END, every SWC_<p>_<h>_1 and TS_<p>_1_1 of the
      chosen profile and every WTD_x_y_z (all from the Meteosens file; SWC in %, WTD in m, negative below the surface:
      the AmeriFlux sign used at US-HB2 / US-Ho1), LE = LE_F_MDS (W/m2, gap-filled) and P_RAIN = P_F (mm per half-hour)
      from the archive's FLUXNET_HH file, matched on TIMESTAMP_START. Missing: -9999.
  <out>_heights.csv: Site_ID, Variable, Height (m; soil depths negative) from VARINFO_METEOSENS, the layout of the
      AmeriFlux measurement-height table.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/icos_to_amf.py --archive data/icos/FR-Bil_ARCHIVE.zip \
        --meteosens data/icos/FR-Bil.zip --site FR-Bil --profile 1 --out output/sites/FR-Bil_amf/FR-Bil
"""
import argparse
import csv
import io
import os
import re
import zipfile

import numpy as np

import plot_site_icos as psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--meteosens', required=True)
    ap.add_argument('--site', required=True)
    ap.add_argument('--profile', default='1')
    ap.add_argument('--out', required=True, help='path prefix')
    args = ap.parse_args()
    za, zm = zipfile.ZipFile(args.archive), zipfile.ZipFile(args.meteosens)
    hts = psi.heights(za)
    msname = [n for n in zm.namelist() if n.lower().endswith('.csv')][0]
    with zm.open(msname) as fh:
        head = next(csv.reader(io.TextIOWrapper(fh, encoding='utf-8')))
    p = args.profile
    swc = sorted([c for c in head if re.fullmatch(rf'SWC_{p}_\d+_1', c)], key=lambda c: -hts.get(c, 0.0))
    ts = [c for c in head if re.fullmatch(rf'TS_{p}_1_1', c)]
    wtd = [c for c in head if re.fullmatch(r'WTD_\d+_\d+_\d+', c)]
    cols = swc + ts + wtd
    tm, vm, _ = psi.read_cols(zm, msname, cols)
    tf, vf, _ = psi.read_cols(za, psi.member(za, r'ICOSETC_[A-Z]{2}-\w{3}_FLUXNET_HH_(INTERIM_)?L2\.csv$'),
                              ['P_F', 'LE_F_MDS'])
    pos = {t: i for i, t in enumerate(tf)}
    j = np.array([pos.get(t, -1) for t in tm])
    flux = {name: np.where(j >= 0, vf[src][np.maximum(j, 0)], np.nan) for name, src in (('LE', 'LE_F_MDS'),
                                                                                         ('P_RAIN', 'P_F'))}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    stamp = lambda t: str(t).replace('-', '').replace('T', '').replace(':', '')[:12]          # noqa: E731
    out_cols = cols + ['LE', 'P_RAIN']
    data = {**vm, **flux}
    with open(args.out + '_BASE_HH.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([f'# Site: {args.site}'])
        w.writerow([f'# Converted from ICOS ETC L2 ({os.path.basename(args.archive)}, {os.path.basename(args.meteosens)})'])
        w.writerow(['TIMESTAMP_START', 'TIMESTAMP_END'] + out_cols)
        for i, t in enumerate(tm):
            row = [stamp(t), stamp(t + np.timedelta64(30, 'm'))]
            for c in out_cols:
                x = data[c][i]
                row.append('-9999' if not np.isfinite(x) else f'{x:.6g}')
            w.writerow(row)
    with open(args.out + '_heights.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Site_ID', 'Variable', 'Height'])
        for c in cols:
            if c in hts:
                w.writerow([args.site, c, hts[c]])
    print(f"[icos2amf] {len(tm)} half-hours {tm[0]} - {tm[-1]}; columns {', '.join(out_cols)}; flux matched on "
          f"{100 * (j >= 0).mean():.0f} %", flush=True)
    for c in swc + wtd:
        print(f"[icos2amf] {c}: height {hts.get(c, np.nan)}", flush=True)
    print(f"[icos2amf] wrote {args.out}_BASE_HH.csv and {args.out}_heights.csv", flush=True)


if __name__ == '__main__':
    main()
