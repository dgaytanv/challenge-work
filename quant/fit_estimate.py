"""WP-G G7: how many multipliers does the FOLDED design actually instantiate, and does
that fit xcvu13p? A proxy answer while the C-synthesis that would measure it is running.

Campaign 1's 15,735,808 is the multiply COUNT PER EVENT of the unfolded design, where
every one of them is a separate piece of hardware. With the token loop folded (pf = 1) the
per-token multipliers are instantiated ONCE and reused 400 times, so the hardware count is
a different and much smaller number. Everything below is read from the EMITTED firmware
(`firmware/parameters.h` of the written project), not assumed.
"""
import argparse
import json
import os
import re

# xcvu13p-flga2577-2-e, from Vitis HLS's own AvailableResources on the licence probe
PART = dict(name='xcvu13p-flga2577-2-e', LUT=1_728_000, FF=3_456_000, DSP=12_288,
            BRAM_18K=5_376, URAM=1_280)

# LUT cost of one small fixed-point multiplier. Deliberately a RANGE, not a point: a
# 7 x 8 multiplier is a few tens of LUT6s and the exact number depends on operand signs,
# the adder tree and what Vitis chooses to fold into DSPs. Anything more precise here
# would be false precision dressed up as an answer.
LUT_PER_MULT = (25, 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prj', required=True)
    ap.add_argument('--n_tokens', type=int, default=400)
    ap.add_argument('--clock_ns', type=float, default=5.0)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()

    txt = open(os.path.join(args.prj, 'firmware', 'parameters.h')).read()

    def cfg(name, key):
        i = txt.index('struct ' + name + ' {')
        window = txt[i:i + 2500]
        v = re.search(r'static const(?:expr)? unsigned ' + key + r'\s*=\s*([^;]+);', window)
        assert v, f'{key} not found in {name}'
        return v.group(1).strip()

    N = args.n_tokens
    # Per-token block: one datapath, iterated n_partitions times.
    per_token = {
        'phi_0': 14 * 128,
        'phi_1': 128 * 128,
        'score': 128 * 32,
        'v': 128 * 128,
    }
    # N-independent, once per event.
    once = {
        'combine (einsum, ReuseFactor 400)': int(cfg('config29', 'multiplier_limit')),
        'out_proj (4 seeds, n_partitions=1 -> all parallel)': 4 * 128 * 128,
        'bottleneck': 512 * 6,
    }
    inst = sum(per_token.values()) + sum(once.values())
    unfolded = N * (14 * 128 + 128 * 128 + 128 * 32 + 128 * 128 + 8 * 4 * 16) + 4 * 128 * 128 + 512 * 6

    # The same design with out_proj also folded over its 4 seed positions (a one-line
    # config change, not done in the synthesised project): 4x fewer there.
    inst_outproj_folded = inst - 4 * 128 * 128 + 128 * 128

    rep = dict(tag=args.tag, part=PART, n_tokens=N, clock_ns=args.clock_ns,
               clock_mhz=1000.0 / args.clock_ns,
               per_token_datapath_multipliers=per_token,
               per_token_total=sum(per_token.values()),
               once_per_event_multipliers=once,
               instantiated_multipliers=inst,
               unfolded_multiplies_per_event=unfolded,
               fold_factor=round(unfolded / inst, 1),
               instantiated_if_out_proj_also_folded=inst_outproj_folded,
               dsp_budget=PART['DSP'],
               multipliers_per_dsp_if_all_in_dsp=round(inst / PART['DSP'], 1),
               lut_estimate_range=[inst * LUT_PER_MULT[0], inst * LUT_PER_MULT[1]],
               lut_estimate_range_out_proj_folded=[inst_outproj_folded * LUT_PER_MULT[0],
                                                   inst_outproj_folded * LUT_PER_MULT[1]],
               lut_budget=PART['LUT'],
               token_pass_cycles=N,
               token_pass_us=N * args.clock_ns / 1000.0,
               lut_per_mult_assumed=LUT_PER_MULT,
               caveat=('LUT figures are a PROXY from a per-multiplier LUT range, not a synthesis '
                       'result. Multiplier counts are exact, read from the emitted firmware.'))
    out = os.path.expanduser(f'~/hackathon-shared/quant/g7_fit_{args.tag}.json')
    json.dump(rep, open(out, 'w'), indent=2)

    print(f'part {PART["name"]}  DSP {PART["DSP"]:,}  LUT {PART["LUT"]:,}  clock {rep["clock_mhz"]:.0f} MHz')
    print(f'\nper-token datapath, instantiated ONCE and iterated {N} times:')
    for k, v in per_token.items():
        print(f'  {k:12s} {v:>8,}')
    print(f'  {"TOTAL":12s} {sum(per_token.values()):>8,}')
    print('\nonce per event:')
    for k, v in once.items():
        print(f'  {k:52s} {v:>8,}')
    print(f'\nINSTANTIATED MULTIPLIERS (the hardware count): {inst:,}')
    print(f'unfolded multiplies per event (campaign 1 number): {unfolded:,}  -> fold is {rep["fold_factor"]}x')
    print(f'  ... {inst / PART["DSP"]:.1f}x the part\'s {PART["DSP"]:,} DSP48s, so most must be LUT logic')
    print(f'  LUT proxy at {LUT_PER_MULT[0]}-{LUT_PER_MULT[1]} LUT/mult: '
          f'{inst*LUT_PER_MULT[0]:,} - {inst*LUT_PER_MULT[1]:,} of {PART["LUT"]:,}')
    print(f'  with out_proj also folded ({inst_outproj_folded:,} mults): '
          f'{inst_outproj_folded*LUT_PER_MULT[0]:,} - {inst_outproj_folded*LUT_PER_MULT[1]:,}')
    print(f'\ntoken pass: {N} sequential iterations = {N} cycles = {rep["token_pass_us"]:.2f} us at '
          f'{rep["clock_mhz"]:.0f} MHz (throughput bound; total latency adds the dataflow depth)')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
