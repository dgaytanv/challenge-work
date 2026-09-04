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
        m0 = re.search(r'^struct ' + name + r'\s*[:{]', txt, re.M)
        assert m0, f'no struct {name}'
        i = m0.start()
        window = txt[i:i + 2500]
        v = re.search(r'static const(?:expr)? unsigned ' + key + r'\s*=\s*([^;]+);', window)
        assert v, f'{key} not found in {name}'
        return v.group(1).strip()

    N = args.n_tokens

    # Read the instantiated multiplier count for each layer from the EMITTED firmware, so
    # this tracks whatever ParallelizationFactor / ReuseFactor the project was built with
    # instead of restating the defaults.
    #   a folded per-token Dense (Conv1D, n_pixels = 1) instantiates n_chan * n_filt / rf
    #   the einsum's allocation is its multiplier_limit
    #   the tail Dense is n_in * n_out / rf
    CONV = {'phi_0': 'config43', 'phi_1': 'config44', 'score': 'config45',
            'v': 'config46', 'out_proj': 'config47'}
    conv_mults, conv_meta = {}, {}
    for lname, cname in CONV.items():
        n_chan = int(cfg(cname, 'n_chan'))
        n_filt = int(cfg(cname, 'n_filt'))
        rf = int(cfg(cname, 'reuse_factor'))
        npart = int(cfg(cname, 'n_partitions'))
        out_w = int(cfg(cname, 'out_width'))
        n_pixels = out_w // npart
        conv_mults[lname] = n_chan * n_filt * n_pixels // rf
        conv_meta[lname] = dict(n_chan=n_chan, n_filt=n_filt, reuse_factor=rf,
                                n_partitions=npart, n_pixels=n_pixels, out_width=out_w,
                                iterations=npart)
    einsum_mults = int(cfg('config29', 'multiplier_limit'))
    bott_rf = int(cfg('config37', 'reuse_factor'))
    bott_mults = 512 * 6 // bott_rf

    per_token = {k: conv_mults[k] for k in ('phi_0', 'phi_1', 'score', 'v')}
    once = {
        'combine (einsum)': einsum_mults,
        f"out_proj (n_partitions={conv_meta['out_proj']['n_partitions']})": conv_mults['out_proj'],
        'bottleneck': bott_mults,
    }
    # cycles: the token pass runs n_partitions iterations, each taking reuse_factor cycles
    token_cycles = max(m['iterations'] * m['reuse_factor']
                       for k, m in conv_meta.items() if k != 'out_proj')
    inst = sum(per_token.values()) + sum(once.values())
    unfolded = N * (14 * 128 + 128 * 128 + 128 * 32 + 128 * 128 + 8 * 4 * 16) + 4 * 128 * 128 + 512 * 6

    # The same design with out_proj also folded over its 4 seed positions (a one-line
    # config change). If it is already folded this is the same number.
    op = conv_meta['out_proj']
    inst_outproj_folded = inst - conv_mults['out_proj'] + conv_mults['out_proj'] // max(op['n_pixels'], 1)

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
               layer_config_from_firmware=conv_meta,
               token_pass_cycles=token_cycles,
               token_pass_us=token_cycles * args.clock_ns / 1000.0,
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
    if inst_outproj_folded != inst:
        print(f'  with out_proj also folded ({inst_outproj_folded:,} mults): '
              f'{inst_outproj_folded*LUT_PER_MULT[0]:,} - {inst_outproj_folded*LUT_PER_MULT[1]:,}')
    else:
        print('  (out_proj is already folded in this configuration)')
    print(f'\ntoken pass: {token_cycles} cycles = {rep["token_pass_us"]:.2f} us at '
          f'{rep["clock_mhz"]:.0f} MHz (throughput bound; total latency adds the dataflow depth)')
    for k, m in conv_meta.items():
        print(f'  {k:10s} n_partitions={m["n_partitions"]:>4} reuse_factor={m["reuse_factor"]} '
              f'-> {conv_mults[k]:>7,} multipliers')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
