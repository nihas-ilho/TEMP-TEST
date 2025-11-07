#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
통합 분석 (최종, '처음 BGR 방식' 복원)
- BGR: INDEX + BGR(행=code, 열=온도 병합 헤더 × [chop_0,chop_1,chop_avg])
       INDEX: chop_avg 기준 (max-min) 최소 code
- MCLK: INDEX + MCLK(행=code, 열=온도); INDEX: (max-min) 최소 code
- WCLK: INDEX + WCLK(행=code, 열=온도); INDEX: (max-min) 최소 code
- TC  : INDEX + TEMP + LED1_current + TC(행=code, 열=온도)
입력: --root 폴더의 모든 *.txt
출력: BGR_code_vs_temp.xlsx / MCLK_code_vs_temp.xlsx / WCLK_code_vs_temp.xlsx / TC_summary.xlsx
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd

# ---------------------- 공통 유틸 ----------------------
RE_TEMP_IN_NAME = re.compile(r'(-?\d+)')

def scan_all_txt(root: Path) -> List[Path]:
    return list(root.rglob('*.txt'))

def pivot_code_vs_temp(per_temp: Dict[int, pd.Series]) -> pd.DataFrame:
    all_codes = sorted({c for s in per_temp.values() for c in s.index})
    temps = sorted(per_temp.keys())
    rows = []
    for code in all_codes:
        row = {'trim_code': code}
        for t in temps:
            row[t] = float(per_temp[t].get(code, pd.NA))
        rows.append(row)
    return pd.DataFrame(rows).set_index('trim_code')[temps]

def compute_stability(matrix: pd.DataFrame, cmin: str, cmax: str, crng: str):
    rec=[]
    for code,row in matrix.iterrows():
        s=pd.to_numeric(row,errors='coerce').dropna()
        if s.empty:
            rec.append({'code':code,'temps':0,cmin:pd.NA,cmax:pd.NA,crng:pd.NA})
        else:
            mn=float(s.min()); mx=float(s.max())
            rec.append({'code':code,'temps':int(s.shape[0]),cmin:mn,cmax:mx,crng:mx-mn})
    idx=pd.DataFrame(rec)
    best_code=best_rng=None
    if not idx.empty:
        v=idx[idx[crng].notna()].sort_values([crng,'code'],kind='mergesort')
        if not v.empty:
            best_code=int(v.iloc[0]['code']); best_rng=float(v.iloc[0][crng])
    return idx,best_code,best_rng

# ---------------------- BGR (처음 방식) ----------------------
# 파일명 예: "-10_BGR_chop_0_Trim_Results_15.txt"
FNAME_BGR = re.compile(
    r'(?P<temp>-?\d+)\s*[_-]\s*BGR\s*[_-]\s*chop\s*[_-]\s*(?P<chop>[01])\s*[_-]\s*Trim\s*[_-]\s*Results\s*[_-]\s*(?P<sample>-?\d+)\.txt',
    re.IGNORECASE
)
# 본문 예: "BGR CODE 12, 1.203 V" 또는 "mV"
LINE_BGR = re.compile(
    r'(?i)\bBGR\s*CODE\s*(?P<code>\d{1,4})\s*[,:\-]\s*(?P<volt>[-+]?\d+(?:\.\d+)?)\s*(?P<u>m?V)?'
)

def _to_volts(val: float, unit: Optional[str]) -> float:
    if unit and unit.lower()=='mv': return float(val)/1000.0
    return float(val)

def parse_bgr_file(path: Path, code_max: int = 4095) -> pd.Series:
    d: Dict[int, float] = {}
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            m = LINE_BGR.search(raw)
            if not m: continue
            code = int(m.group('code'))
            if not (0 <= code <= code_max): continue
            v = _to_volts(float(m.group('volt')), m.group('u'))
            d[code] = v
    return pd.Series(d, dtype='float64').sort_index()

def collect_bgr_by_name(root: Path) -> Dict[Tuple[int,int], pd.Series]:
    per: Dict[Tuple[int,int], pd.Series] = {}
    for p in scan_all_txt(root):
        m = FNAME_BGR.search(p.name)
        if not m: continue
        temp = int(m.group('temp'))
        chop = int(m.group('chop'))
        s = parse_bgr_file(p)
        if not s.empty:
            per[(temp, chop)] = s
    return per

def save_bgr(bgr_map: Dict[Tuple[int,int], pd.Series], out_path: Path):
    if not bgr_map:
        print("BGR: no parsed data")
        return
    temps = sorted({t for (t, ch) in bgr_map.keys()})
    codes = sorted({c for s in bgr_map.values() for c in s.index})

    # 멀티컬럼(상위=temp, 하위=mode)
    top=[]; sub=[]
    for t in temps:
        top += [t,t,t]
        sub += ['chop_0','chop_1','chop_avg']
    cols = pd.MultiIndex.from_arrays([top,sub], names=['temp','mode'])

    rows=[]
    for code in codes:
        row={'trim_code': code}
        for t in temps:
            v0 = bgr_map.get((t,0), pd.Series(dtype='float64')).get(code, pd.NA)
            v1 = bgr_map.get((t,1), pd.Series(dtype='float64')).get(code, pd.NA)
            vals=[x for x in (v0,v1) if pd.notna(x)]
            vavg=float(sum(vals)/len(vals)) if vals else pd.NA
            row[(t,'chop_0')] = float(v0) if pd.notna(v0) else pd.NA
            row[(t,'chop_1')] = float(v1) if pd.notna(v1) else pd.NA
            row[(t,'chop_avg')] = vavg
        rows.append(row)
    df = pd.DataFrame(rows).set_index('trim_code').reindex(columns=cols)

    # 안정성 분석: chop_0, chop_1, chop_avg 각각 수행
    chop0 = df.xs(key='chop_0', axis=1, level='mode')
    idx_0, best_code_0, best_rng_0 = compute_stability(chop0, 'BGR0_min','BGR0_max','BGR0_range')

    chop1 = df.xs(key='chop_1', axis=1, level='mode')
    idx_1, best_code_1, best_rng_1 = compute_stability(chop1, 'BGR1_min','BGR1_max','BGR1_range')

    avg = df.xs(key='chop_avg', axis=1, level='mode')
    idx_avg, best_code_avg, best_rng_avg = compute_stability(avg, 'BGRavg_min','BGRavg_max','BGRavg_range')

    # INDEX 시트용 데이터프레임은 avg 기준으로만 생성
    final_idx = idx_avg.copy()
    if not final_idx.empty:
        final_idx.insert(0, 'HEX', final_idx['code'].apply(lambda x: f'{x:X}'))


    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as w:
        # INDEX 시트 상단 요약 정보 (chop_avg 기준)
        summary_data = {
            'num_codes': df.shape[0],
            'num_temps': len(temps),
            'most_stable_code_by_chop_avg': best_code_avg if best_code_avg is not None else 'N/A',
            'BGRavg_range_V': best_rng_avg if best_rng_avg is not None else 'N/A',
        }
        pd.DataFrame([summary_data]).to_excel(w, 'INDEX', index=False, startrow=0)

        # INDEX 시트 안정성 테이블 (chop_avg 기준)
        if not final_idx.empty:
            # BGRavg_range 기준으로 정렬
            idx2 = final_idx.sort_values(['BGRavg_range','code'], na_position='last', kind='mergesort')
            
            # HEX가 code 왼쪽에 오도록 열 순서 재정렬
            ordered_cols = ['HEX', 'code']
            for c in idx2.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)
            idx2 = idx2[ordered_cols]

            cols_to_round = ['BGRavg_min', 'BGRavg_max', 'BGRavg_range']
            for c in cols_to_round:
                if c in idx2.columns:
                    idx2[c] = pd.to_numeric(idx2[c], errors='coerce').round(6)
            idx2.to_excel(w, 'INDEX', index=False, startrow=4)

        # BGR 시트(온도 병합 헤더 수작업)
        ws = w.book.add_worksheet('BGR')
        fmt_top = w.book.add_format({'align':'center','valign':'vcenter','bold':True,'border':1})
        fmt_sub = w.book.add_format({'align':'center','bold':True,'border':1})
        fmt_idx = w.book.add_format({'align':'center','bold':True,'border':1,'bg_color':'#F2F2F2'})
        fmt_cell= w.book.add_format({'border':1})

        r0,r1=0,1; c=0
        ws.merge_range(r0,c,r1,c,'HEX',fmt_idx); c+=1
        ws.merge_range(r0,c,r1,c,'trim_code',fmt_idx); c+=1
        for t in temps:
            ws.merge_range(r0, c, r0, c+2, f"T{t}C", fmt_top)
            ws.write(r1, c+0, 'chop_0', fmt_sub)
            ws.write(r1, c+1, 'chop_1', fmt_sub)
            ws.write(r1, c+2, 'chop_avg', fmt_sub)
            c += 3

        r=2
        for code, row in df.iterrows():
            ws.write(r, 0, f'{int(code):X}', fmt_cell)
            ws.write(r, 1, int(code), fmt_cell)
            cc=2
            for t in temps:
                for subcol in ('chop_0','chop_1','chop_avg'):
                    val=row[(t, subcol)]
                    ws.write(r, cc, float(val) if pd.notna(val) else None, fmt_cell)
                    cc += 1
            r += 1

        ws.set_column('A:A', 10)
        ws.set_column('B:B', 12)
        ws.set_column(2, 2+len(temps)*3-1, 11)

        # chop_0, chop_1, chop_avg 시트 추가
        def write_chop_sheet(df_chop, sheet_name):
            if not df_chop.empty:
                df_to_write = df_chop.reset_index()
                df_to_write.insert(0, 'HEX', df_to_write['trim_code'].apply(lambda x: f'{x:X}'))
                cols = ['HEX', 'trim_code'] + [c for c in df_to_write.columns if c not in ['HEX', 'trim_code']]
                df_to_write = df_to_write[cols]
                df_to_write.to_excel(w, sheet_name, index=False)

        write_chop_sheet(chop0, 'chop_0')
        write_chop_sheet(chop1, 'chop_1')
        write_chop_sheet(avg, 'chop_avg')

        # --- 차트 추가 ---
        from xlsxwriter.utility import xl_col_to_name
        
        def add_bgr_chart(writer, sheet_name, df_data, best_code_val, num_temps_val, nvm_code_val=None):
            if df_data.empty or num_temps_val == 0:
                return

            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
            last_col = xl_col_to_name(2 + num_temps_val) # HEX, trim_code, temps...
            x_values = f"='{sheet_name}'!$C$1:${last_col}$1"

            # Series 1: 최적코드 (Best Code)
            if best_code_val is not None:
                try:
                    # df_data is indexed by trim_code (integer)
                    row_num_best = df_data.index.get_loc(best_code_val) + 2
                    y_values_best = f"='{sheet_name}'!$C${row_num_best}:${last_col}${row_num_best}"
                    chart.add_series({
                        'name':       f'최적코드({hex(best_code_val).upper()[2:]})',
                        'categories': x_values,
                        'values':     y_values_best,
                    })
                except KeyError:
                    print(f"Warning: best code '{best_code_val}' not found for charting in sheet '{sheet_name}'.")

            # Series 2: NVM 코드 (Fixed Code)
            if nvm_code_val is not None:
                try:
                    row_num_nvm = df_data.index.get_loc(nvm_code_val) + 2
                    y_values_nvm = f"='{sheet_name}'!$C${row_num_nvm}:${last_col}${row_num_nvm}"
                    chart.add_series({
                        'name':       f'NVM({hex(nvm_code_val).upper()[2:]})',
                        'categories': x_values,
                        'values':     y_values_nvm,
                    })
                except KeyError:
                    print(f"Warning: NVM code '{nvm_code_val}' not found in sheet '{sheet_name}' index for charting.")

            # Set titles and labels
            if sheet_name == 'chop_avg':
                 chart.set_title({'name': 'BGR Voltage (chop 평균)'})
            else:
                 chart.set_title({'name': f'BGR Voltage ({sheet_name})'})

            chart.set_x_axis({'name': 'TEMP [°C]'})
            chart.set_y_axis({'name': 'Voltage [V]'})
            chart.set_style(3)
            chart.set_legend({'position': 'bottom'})

            insert_col_letter = xl_col_to_name(2 + num_temps_val + 2)
            worksheet.insert_chart(f'{insert_col_letter}2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

        # 차트 생성 호출
        nvm_code = 90 # 0x5A
        add_bgr_chart(w, 'chop_0', chop0, best_code_0, len(temps), nvm_code_val=nvm_code)
        add_bgr_chart(w, 'chop_1', chop1, best_code_1, len(temps), nvm_code_val=nvm_code)
        add_bgr_chart(w, 'chop_avg', avg, best_code_avg, len(temps), nvm_code_val=nvm_code)

    print(f"✅ BGR 저장: {out_path.resolve()}")

# ---------------------- CLK (MCLK/WCLK) ----------------------
RE_CLK_LINE = re.compile(
    r'(?i)\b(?P<clk>MCLK|WCLK)\s*CODE\s*(?P<code>\d{1,4})\s*[,:\-]\s*(?P<f>[-+]?\d+(?:\.\d+)?)\s*(?P<u>[kKmM]?[hH]z)?'
)

def _to_khz(val: float, unit: Optional[str]) -> float:
    if not unit: return float(val)        # 기본 kHz
    u=unit.lower()
    if u=='hz':  return float(val)/1000.0
    if u=='mhz': return float(val)*1000.0
    return float(val)                     # kHz

def collect_clk(root: Path, target: str) -> Dict[int, pd.Series]:
    per: Dict[int, pd.Series] = {}
    for p in scan_all_txt(root):
        mtemp = RE_TEMP_IN_NAME.search(p.name)
        if not mtemp: continue
        temp = int(mtemp.group(1))
        d: Dict[int, float] = {}
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                m = RE_CLK_LINE.search(raw)
                if not m: continue
                if m.group('clk').upper()!=target.upper(): continue
                code=int(m.group('code'))
                freq=float(m.group('f')); unit=m.group('u')
                d[code]=_to_khz(freq, unit)
        if d:
            per[temp] = pd.Series(d, dtype='float64').sort_index()
    return per

def save_clk(clk_map: Dict[int, pd.Series], target: str, out_path: Path):
    if not clk_map:
        print(f"{target}: no parsed data")
        return
    mat = pivot_code_vs_temp(clk_map)
    idx, best_code, best_rng = compute_stability(mat, f'{target}_min', f'{target}_max', f'{target}_range')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as w:
        pd.DataFrame([{
            'most_stable_code': best_code if best_code is not None else 'N/A',
            f'{target}_range_kHz': best_rng if best_rng is not None else 'N/A',
            'num_codes': mat.shape[0],
            'num_temps': mat.shape[1],
        }]).to_excel(w, 'INDEX', index=False, startrow=0)
        if not idx.empty:
            idx.insert(0, 'HEX', idx['code'].apply(lambda x: f'{x:X}'))
            idx2 = idx.sort_values([f'{target}_range','code'], na_position='last', kind='mergesort')
            
            ordered_cols = ['HEX', 'code']
            for c in idx2.columns:
                if c not in ordered_cols:
                    ordered_cols.append(c)
            idx2 = idx2[ordered_cols]

            for c in [f'{target}_min', f'{target}_max', f'{target}_range']:
                if c in idx2.columns:
                    idx2[c] = pd.to_numeric(idx2[c], errors='coerce').round(6)
            idx2.to_excel(w, 'INDEX', index=False, startrow=4)
        
        mat_to_write = mat.reset_index()
        mat_to_write.insert(0, 'HEX', mat_to_write['trim_code'].apply(lambda x: f'{x:X}'))
        cols = ['HEX', 'trim_code'] + [c for c in mat_to_write.columns if c not in ['HEX', 'trim_code']]
        mat_to_write = mat_to_write[cols]
        mat_to_write.to_excel(w, target.upper(), index=False)

    print(f"✅ {target} 저장: {out_path.resolve()}")

# ---------------------- TC ----------------------
RE_TC_LINE   = re.compile(r'(?i)\bTC\s*CODE\s*(?P<code>\d{1,4})\s*[,:\-]\s*(?P<i>[-+]?\d+(?:\.\d+)?)\s*mA')
RE_TEMP_TOP  = re.compile(r'(?i)^\s*TEMP\s*=\s*(?P<raw>\d+)')
RE_LED_TOP   = re.compile(r'(?i)^\s*LED1[_ ]?Current\s*=\s*(?P<val>[-+]?\d+(?:\.\d+)?)\s*mA')

def collect_tc(root: Path):
    tc_map: Dict[int, Dict[int, float]] = {}
    temp_rows: List[Tuple[int, Optional[int]]] = []
    led_rows:  List[Tuple[int, Optional[float]]] = []
    for p in scan_all_txt(root):
        mtemp = RE_TEMP_IN_NAME.search(p.name)
        if not mtemp: continue
        temp = int(mtemp.group(1))
        d: Dict[int, float]={}
        temp_raw=None; led_ma=None; found=False
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                s=raw.strip()
                m=RE_TC_LINE.search(s)
                if m:
                    code=int(m.group('code')); curr=float(m.group('i'))
                    d[code]=curr; found=True
                    continue
                if temp_raw is None:
                    m=RE_TEMP_TOP.search(s)
                    if m:
                        try: temp_raw=int(m.group('raw'))
                        except: pass
                if led_ma is None:
                    m=RE_LED_TOP.search(s)
                    if m:
                        try: led_ma=float(m.group('val'))
                        except: pass
        if found:
            tc_map.setdefault(temp,{}).update(d)
            temp_rows.append((temp, temp_raw))
            led_rows.append((temp, led_ma))
    temp_df = (pd.DataFrame(temp_rows, columns=['temp','TEMP_raw'])
               .drop_duplicates('temp', keep='last').set_index('temp').sort_index()) if temp_rows else pd.DataFrame(columns=['TEMP_raw'])
    led_df  = (pd.DataFrame(led_rows, columns=['temp','LED1_Current_mA'])
               .drop_duplicates('temp', keep='last').set_index('temp').sort_index()) if led_rows else pd.DataFrame(columns=['LED1_Current_mA'])
    return temp_df, led_df, tc_map

def save_tc(tc_temp_df: pd.DataFrame, tc_led_df: pd.DataFrame, tc_map: Dict[int, Dict[int, float]], out_path: Path):
    if not tc_map and tc_temp_df.empty and tc_led_df.empty:
        print("TC: no files")
        return
    temps = sorted(tc_map.keys())
    all_codes = sorted({c for d in tc_map.values() for c in d.keys()})
    rows=[]
    for code in all_codes:
        row={'trim_code': code}
        for t in temps:
            row[t]=tc_map.get(t,{}).get(code, pd.NA)
        rows.append(row)
    tc_df = pd.DataFrame(rows).set_index('trim_code')[temps] if rows else pd.DataFrame()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as w:
        summary_df = pd.DataFrame([{'num_temps': len(temps), 'num_codes': (tc_df.shape[0] if not tc_df.empty else 0)}])
        summary_df.to_excel(w, 'INDEX', index=False)
        
        # 데이터 시트 쓰기
        temp_sheet_df = tc_temp_df if not tc_temp_df.empty else pd.DataFrame(columns=['TEMP_raw'])
        temp_sheet_df.to_excel(w, 'TEMP')
        
        led_sheet_df = tc_led_df if not tc_led_df.empty else pd.DataFrame(columns=['LED1_Current_mA'])
        led_sheet_df.to_excel(w, 'LED1_current')
        
        if not tc_df.empty:
            tc_df_to_write = tc_df.reset_index()
            tc_df_to_write.insert(0, 'HEX', tc_df_to_write['trim_code'].apply(lambda x: f'{x:X}'))
            cols = ['HEX', 'trim_code'] + [c for c in tc_df_to_write.columns if c not in ['HEX', 'trim_code']]
            tc_df_to_write = tc_df_to_write[cols]
            tc_df_to_write.to_excel(w, 'TC', index=False)
        else:
            pd.DataFrame(columns=['HEX', 'trim_code']).to_excel(w, 'TC', index=False)

        # --- TC 관련 시트에 차트 추가 ---
        def add_tc_chart(writer, sheet_name, df_data, y_col_name, y_axis_title, chart_title):
            if df_data.empty:
                return
            
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            chart = workbook.add_chart({'type': 'scatter', 'subtype': 'smooth_with_markers'})
            
            num_points = df_data.shape[0]
            # X 값은 인덱스(A열), Y 값은 첫 번째 데이터 열(B열)에 있습니다.
            x_values = f"='{sheet_name}'!$A$2:$A${1 + num_points}"
            y_values = f"='{sheet_name}'!$B$2:$B${1 + num_points}"

            chart.add_series({
                'name':       y_col_name,
                'categories': x_values,
                'values':     y_values,
            })

            chart.set_title({'name': chart_title})
            chart.set_x_axis({'name': 'Temperature (°C)'})
            chart.set_y_axis({'name': y_axis_title})
            chart.set_style(3)
            chart.set_legend({'position': 'none'}) # 단일 계열이므로 범례는 필요 없음

            worksheet.insert_chart('D2', chart)

        # TEMP 시트에 차트 추가
        if not tc_temp_df.empty:
            add_tc_chart(w, 'TEMP', tc_temp_df, 
                         y_col_name='TEMP_raw', 
                         y_axis_title='Raw Value', 
                         chart_title='TEMP Raw Value vs Temperature')

        # LED1_current 시트에 차트 추가
        if not tc_led_df.empty:
            add_tc_chart(w, 'LED1_current', tc_led_df, 
                         y_col_name='LED1_Current_mA', 
                         y_axis_title='Current (mA)', 
                         chart_title='LED1 Current vs Temperature')

    print(f"✅ TC 저장: {out_path.resolve()}")

# ---------------------- MAIN ----------------------
def main():
    ap = argparse.ArgumentParser(description="BGR/MCLK/WCLK/TC 통합 분석 (초기 BGR 파서 복원)")
    ap.add_argument('--root', type=Path, default=Path('DATA'), help='데이터 폴더 (기본: ./DATA)')
    args = ap.parse_args()

    root = args.root

    # BGR (초기 코드 방식)
    bgr_map = collect_bgr_by_name(root)
    save_bgr(bgr_map, Path('BGR_code_vs_temp.xlsx'))

    # MCLK / WCLK
    mclk_map = collect_clk(root, 'MCLK')
    wclk_map = collect_clk(root, 'WCLK')
    save_clk(mclk_map, 'MCLK', Path('MCLK_code_vs_temp.xlsx'))
    save_clk(wclk_map, 'WCLK', Path('WCLK_code_vs_temp.xlsx'))

    # TC
    tc_temp_df, tc_led_df, tc_map = collect_tc(root)
    save_tc(tc_temp_df, tc_led_df, tc_map, Path('TC_summary.xlsx'))

if __name__ == '__main__':
    main()
