import os, json, csv
from statistics import mean
from cavp.config import Cfg
from cavp.training import train_validate_one

def sweep():
    out_dir_log = "logs/T30_v3A_Attempt3"
    out_dir_ckpt = "ckpts/T30_v3A_Attempt3"
    os.makedirs(out_dir_log, exist_ok=True); os.makedirs(out_dir_ckpt, exist_ok=True)

    VARIANTS = [
        ("T30_v3A_full",             dict(enable_video=True,  use_future_video=True,  enable_cv_gain=True)),
        ("T30_v3A_hist_only_video",  dict(enable_video=True,  use_future_video=False, enable_cv_gain=True)),
        ("T30_v3A_no_cv_gain",       dict(enable_video=True,  use_future_video=True,  enable_cv_gain=False)),
        ("T30_v3A_no_video",         dict(enable_video=False, use_future_video=False, enable_cv_gain=False)),
    ]
    SEEDS = [7, 17, 42]

    all_results=[]; flat_rows=[]
    CSV_HEADER = [
        "variant","seed","top1_all","md_all",
        "top1_01","md_01","top1_02","md_02","top1_03","md_03",
        "best_epoch","seg_01","seg_02","seg_03"
    ]

    for vname, overrides in VARIANTS:
        for sd in SEEDS:
            cfg = Cfg()
            assert cfg.T_future>=30 and cfg.process_frame_nums>=60
            for k,v in overrides.items(): setattr(cfg, k, v)
            cfg.seed = sd

            res = train_validate_one(cfg, vname, out_dir_ckpt)
            all_results.append(res)

            va = res["best_val"]; seg = va["seg"]
            flat_rows.append([
                vname, sd,
                f'{va["top1_all"]:.6f}', f'{va["md_all"]:.6f}',
                f'{va["t1"]:.6f}', f'{va["m1"]:.6f}',
                f'{va["t2"]:.6f}', f'{va["m2"]:.6f}',
                f'{va["t3"]:.6f}', f'{va["m3"]:.6f}',
                res["best_epoch"], seg[0], seg[1], seg[2]
            ])

    with open(os.path.join(out_dir_log, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    flat_csv = os.path.join(out_dir_log, "sweep_flat.csv")
    with open(flat_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(CSV_HEADER); w.writerows(flat_rows)
    print(f"[LOG] Wrote flat results to {flat_csv}")

    # Aggregate mean±std (population)
    agg = {}
    for row in flat_rows:
        vname=row[0]
        if vname not in agg: agg[vname] = {"top1_all":[], "md_all":[], "t1":[], "m1":[], "t2":[], "m2":[], "t3":[], "m3":[]}
        agg[vname]["top1_all"].append(float(row[2])); agg[vname]["md_all"].append(float(row[3]))
        agg[vname]["t1"].append(float(row[4])); agg[vname]["m1"].append(float(row[5]))
        agg[vname]["t2"].append(float(row[6])); agg[vname]["m2"].append(float(row[7]))
        agg[vname]["t3"].append(float(row[8])); agg[vname]["m3"].append(float(row[9]))

    def m_s(a):
        m=sum(a)/len(a); s=(sum((x-m)**2 for x in a)/len(a))**0.5
        return f"{m:.4f}", f"{s:.4f}"

    agg_csv = os.path.join(out_dir_log, "sweep_agg.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["variant",
                    "top1_mean","top1_std","md_mean","md_std",
                    "top1_01_mean","top1_01_std","md_01_mean","md_01_std",
                    "top1_02_mean","top1_02_std","md_02_mean","md_02_std",
                    "top1_03_mean","top1_03_std","md_03_mean","md_03_std"])
        for vname, d in agg.items():
            t1m,t1s = m_s(d["top1_all"]); mdm,mds = m_s(d["md_all"])
            a1m,a1s = m_s(d["t1"]); m1m,m1s = m_s(d["m1"])
            a2m,a2s = m_s(d["t2"]); m2m,m2s = m_s(d["m2"])
            a3m,a3s = m_s(d["t3"]); m3m,m3s = m_s(d["m3"])
            w.writerow([vname, t1m,t1s, mdm,mds,
                        a1m,a1s, m1m,m1s,
                        a2m,a2s, m2m,m2s,
                        a3m,a3s, m3m,m3s])
    print(f"[LOG] Wrote aggregated results to {agg_csv}")

if __name__ == "__main__":
    sweep()
