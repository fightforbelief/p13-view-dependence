"""Render paired width differences from the checked Windows results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent
rows = json.loads((root.parent / "historical_results.json").read_text())["origins"]
fig,ax = plt.subplots(figsize=(6.7,2.6),layout="constrained")
for r in rows:
    def width(key,cap):
        lo,hi=(sum(pair)*50 for pair in r[key])
        if cap:
            lo,hi=(min(20,max(-20,x)) for x in (lo,hi))
        return hi-lo
    raw,limited=(max(0,width("full",cap)-width("signed",cap)) for cap in (False,True))
    ax.plot([r["year"]]*2,[limited,raw],color="#a2b1be",lw=1.2,zorder=1)
    ax.scatter(r["year"],raw,color="#264f77",s=12,zorder=3)
    ax.scatter(r["year"],limited,marker="x",color="#ae4f28",s=15,linewidth=.95,zorder=4)
ax.set(xlabel="January decision year",ylabel="Additional width (pp)",xlim=(1978.5,2026.5),ylim=(-.8,21.8))
ax.axhline(0,color="0.72",lw=.6)
ax.spines[["right","top"]].set_visible(False)
ax.legend(handles=[Line2D([],[],linestyle="",marker="o",color="#264f77",label="Unrestricted position"),
                   Line2D([],[],linestyle="",marker="x",color="#ae4f28",label="Manufacturing weight limited to 0–40%")],
          frameon=False,loc="upper right",fontsize=8)
fig.savefig(root / "widths.pdf")
