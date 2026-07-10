#!/usr/bin/env python3
"""Refine mapping of human modified rows to MOD_SC IDs and produce an audit CSV.

Outputs:
 - outputs/human_modifier_mapping_audit.csv
 - outputs/human_modifier_flips_refined.csv

Strategy:
 - Load canonical modifiers from Dataset/modifiers_batch1.json
 - For each human modified row (is_modified==1), try in order:
   1) exact scenario_id + axis match
   2) exact scenario_id + modifier_text match
   3) order-based match within (participant_id, theme_id) using modifier ordering per scenario
   4) fuzzy substring match of modifier_text within scenario modifiers
 - Record mapping attempts and confidence score in audit CSV
"""

import csv
import json
import os
from collections import defaultdict
from difflib import SequenceMatcher

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'outputs')
os.makedirs(OUT, exist_ok=True)

MOD_FILE = os.path.join(ROOT, 'Dataset', 'modifiers_batch1.json')
HUMAN_CSV = os.path.join(ROOT, 'human_study', 'analysis_ready.csv')
EXISTING = os.path.join(OUT, 'human_modifier_flips.csv')


def load_mods():
    with open(MOD_FILE) as f:
        mods = json.load(f)
    by_s = {s['scenario_id']: s['modifiers'] for s in mods}
    return by_s


def similar(a,b):
    return SequenceMatcher(None,a,b).ratio()


def main():
    mods = load_mods()
    # load existing mapping for reference
    existing = {}
    if os.path.exists(EXISTING):
        with open(EXISTING) as f:
            r = csv.DictReader(f)
            for row in r:
                existing[(row['participant_id'], row['modifier_id'])] = int(row['flip'])

    audit_rows = []
    refined = []
    with open(HUMAN_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                is_mod = int(row.get('is_modified','0'))
            except Exception:
                is_mod = 0
            if is_mod == 0:
                continue
            pid = row.get('participant_id')
            sc = row.get('scenario_id')
            theme = row.get('theme_id')
            axis = (row.get('axis') or '').strip()
            mtext = (row.get('modifier_text') or '').strip()
            choice = row.get('choice')

            mapped = None
            method = None
            score = 0.0

            # 1) exact scenario + axis
            if sc and sc in mods:
                for m in mods[sc]:
                    if m.get('axis') == axis and axis!='':
                        mapped = m['modifier_id']; method='scenario_axis'; score=1.0; break

            # 2) exact scenario + text
            if mapped is None and sc and sc in mods and mtext:
                for m in mods[sc]:
                    if m.get('modifier_text','').strip()==mtext:
                        mapped = m['modifier_id']; method='scenario_text'; score=1.0; break

            # 3) order-based: take modifier position by axis frequency within theme group
            if mapped is None and sc and sc in mods:
                # if axis empty, skip
                if axis:
                    # attempt to find by axis order index across scenario modifiers
                    for idx,m in enumerate(mods[sc]):
                        if m.get('axis')==axis:
                            mapped = m['modifier_id']; method='scenario_axis_index'; score=0.9; break

            # 4) fuzzy text match within scenario
            if mapped is None and sc and sc in mods and mtext:
                best=None; best_score=0.0
                for m in mods[sc]:
                    s = similar(mtext.lower(), m.get('modifier_text','').lower())
                    if s>best_score:
                        best_score=s; best=m
                if best_score>0.6:
                    mapped = best['modifier_id']; method='fuzzy_text'; score=best_score

            audit_rows.append({'participant_id':pid,'theme_id':theme,'scenario_id':sc,'axis':axis,'modifier_text':mtext,'mapped_modifier':mapped or '', 'method':method or '', 'score':score})

            if mapped:
                # determine flip compared to baseline within participant+theme
                # look for baseline row in human csv by scanning file again (simple but okay given size)
                flip = 0
                # We'll recompute baseline by reading the file again
                # (cheap) - find first baseline for pid+theme
                with open(HUMAN_CSV) as f2:
                    r2 = csv.DictReader(f2)
                    baseline_choice=None
                    for rrr in r2:
                        if rrr.get('participant_id')==pid and rrr.get('theme_id')==theme and rrr.get('is_modified','0') in ['0','']:
                            baseline_choice = rrr.get('choice'); break
                    try:
                        flip = 1 if str(choice) != str(baseline_choice) else 0
                    except Exception:
                        flip = 0
                refined.append({'participant_id':pid,'modifier_id':mapped,'flip':flip})

    # write audit
    with open(os.path.join(OUT,'human_modifier_mapping_audit.csv'),'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['participant_id','theme_id','scenario_id','axis','modifier_text','mapped_modifier','method','score'])
        w.writeheader()
        for r in audit_rows:
            w.writerow(r)

    # write refined flips
    with open(os.path.join(OUT,'human_modifier_flips_refined.csv'),'w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['participant_id','modifier_id','flip'])
        w.writeheader()
        for r in refined:
            w.writerow(r)

    print('Wrote audit and refined mapping outputs:', os.path.join(OUT,'human_modifier_mapping_audit.csv'), os.path.join(OUT,'human_modifier_flips_refined.csv'))


if __name__=='__main__':
    main()
