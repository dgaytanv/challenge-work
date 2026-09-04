"""WP-G G7: turn a Vitis HLS csynth report (and a Vivado utilisation report) into JSON.

Every number in the writeup's Synthesis table comes from here, not from a hand copy, and
the source file for each is recorded next to it. Percentages are computed against the
tool's OWN 'Available' figures for the part, so the part never has to be looked up.
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET


def parse_csynth_xml(path):
    r = ET.parse(path).getroot()

    def g(p, cast=str):
        e = r.find(p)
        if e is None or e.text is None:
            return None
        t = e.text.strip()
        try:
            return cast(t)
        except (TypeError, ValueError):
            return t

    out = {'source': path}
    out['part'] = g('./UserAssignments/Part')
    out['top'] = g('./UserAssignments/TopModelName')
    out['target_clock_ns'] = g('./UserAssignments/TargetClockPeriod', float)
    out['clock_uncertainty_ns'] = g('./UserAssignments/ClockUncertainty', float)
    out['estimated_clock_ns'] = g('./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod', float)
    if out['estimated_clock_ns']:
        out['estimated_fmax_mhz'] = 1000.0 / out['estimated_clock_ns']
    if out['target_clock_ns']:
        out['target_clock_mhz'] = 1000.0 / out['target_clock_ns']

    lat = './PerformanceEstimates/SummaryOfOverallLatency/'
    for k, tag in (('latency_best_cycles', 'Best-caseLatency'),
                   ('latency_worst_cycles', 'Worst-caseLatency'),
                   ('latency_avg_cycles', 'Average-caseLatency'),
                   ('ii_best', 'Best-caseII'), ('ii_worst', 'Worst-caseII'),
                   ('interval_min_cycles', 'Interval-min'), ('interval_max_cycles', 'Interval-max'),
                   ('pipeline_type', 'PipelineType')):
        out[k] = g(lat + tag, int if 'Type' not in tag else str)
    # latency in microseconds at the TARGET clock, which is the number a trigger cares about
    if out.get('latency_worst_cycles') and out.get('target_clock_ns'):
        out['latency_worst_us'] = out['latency_worst_cycles'] * out['target_clock_ns'] / 1000.0
    if out.get('interval_max_cycles') and out.get('target_clock_ns'):
        out['ii_max_cycles'] = out['interval_max_cycles']
        out['ii_max_us'] = out['interval_max_cycles'] * out['target_clock_ns'] / 1000.0

    res = './AreaEstimates/Resources/'
    avail = './AreaEstimates/AvailableResources/'
    used, available, pct = {}, {}, {}
    for k in ('LUT', 'FF', 'DSP', 'BRAM_18K', 'URAM'):
        u, a = g(res + k, int), g(avail + k, int)
        if u is None:
            continue
        used[k] = u
        available[k] = a
        pct[k] = round(100.0 * u / a, 3) if a else None
    out['used'] = used
    out['available'] = available
    out['pct_of_part'] = pct
    return out


UTIL_ROWS = (('LUT', r'CLB LUTs'), ('FF', r'CLB Registers'), ('DSP', r'DSPs'),
             ('BRAM36', r'Block RAM Tile'), ('URAM', r'URAM'))


def parse_vivado_util(path):
    """Parse `report_utilization` text (the OOC post-synthesis report)."""
    txt = open(path, errors='replace').read()
    out = {'source': path, 'used': {}, 'available': {}, 'pct_of_part': {}}
    for key, label in UTIL_ROWS:
        m = re.search(r'^\|\s*' + label + r'\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)?\s*\|\s*([\d.]+)?\s*\|'
                      r'\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|', txt, re.M)
        if not m:
            continue
        out['used'][key] = float(m.group(1))
        out['available'][key] = float(m.group(4))
        out['pct_of_part'][key] = float(m.group(5))
    return out


def parse_vivado_timing(path):
    """Worst negative slack from a `report_timing_summary` text report."""
    txt = open(path, errors='replace').read()
    out = {'source': path}
    for key, label in (('wns_ns', 'WNS'), ('tns_ns', 'TNS'), ('whs_ns', 'WHS')):
        m = re.search(label + r'\(ns\)[^\n]*\n[-\s]*\n\s*(-?[\d.]+)', txt)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r'^\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(\d+)\s+(\d+)\s+(-?[\d.]+)', txt, re.M)
    if m and 'wns_ns' not in out:
        out['wns_ns'] = float(m.group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prj', required=True, help='hls4ml project directory')
    ap.add_argument('--project_name', default='pma0')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--extra', default=None, help='JSON string of extra fields to record')
    args = ap.parse_args()

    rep = {'tag': args.tag, 'prj': args.prj}
    if args.extra:
        rep.update(json.loads(args.extra))

    sol = os.path.join(args.prj, f'{args.project_name}_prj', 'solution1')
    xml = os.path.join(sol, 'syn', 'report', f'{args.project_name}_csynth.xml')
    if os.path.exists(xml):
        rep['csynth'] = parse_csynth_xml(xml)
    else:
        rep['csynth'] = f'MISSING: {xml}'

    cosim = glob.glob(os.path.join(sol, 'sim', 'report', '*_cosim.rpt'))
    if cosim:
        rep['cosim_rpt'] = open(cosim[0], errors='replace').read()[:4000]

    for name, fn in (('vivado_util', parse_vivado_util), ('vivado_timing', parse_vivado_timing)):
        for cand in (os.path.join(args.prj, f'{name}.rpt'),
                     os.path.join(args.prj, 'vivado_synth.rpt'),
                     os.path.join(args.prj, f'{args.project_name}_vivado_synth.rpt')):
            if os.path.exists(cand):
                try:
                    rep[name] = fn(cand)
                except Exception as e:
                    rep[name] = f'{type(e).__name__}: {e}'
                break

    out = os.path.expanduser(f'~/hackathon-shared/quant/g7_{args.tag}.json')
    with open(out, 'w') as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep.get('csynth', {}), indent=2)[:2500])
    print(f'[parse] wrote {out}')


if __name__ == '__main__':
    sys.exit(main() or 0)
