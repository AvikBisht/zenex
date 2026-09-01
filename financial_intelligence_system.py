"""
financial_intelligence_system.py  (v2 - enhanced visuals)
PS-01 - Multi-Agent Autonomous Financial Intelligence System for Retail Investors
HackVerse: Into The Web -- Sprint 1 (Rapid Vibe Coding)

Run:  python financial_intelligence_system.py
Needs: pip install customtkinter yfinance matplotlib
"""

import customtkinter as ctk
import threading
import time
import random
import hashlib
import statistics
import csv
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# ---------------------------------------------------------------------------
# SYNTHETIC DATA LAYER
# ---------------------------------------------------------------------------

SECTOR_MAP = {
    "RELIANCE.NS": "Energy/Retail",
    "TCS.NS": "IT Services",
    "INFY.NS": "IT Services",
    "HDFCBANK.NS": "Banking",
    "TATAMOTORS.NS": "Auto",
}

DOCUMENT_CORPUS = [
    {"id": "DOC-SEBI-2026-014", "ticker": "RELIANCE.NS",
     "text": "Reliance Industries quarterly filing reports strong retail segment growth, "
             "with expansion into digital commerce. Capex remains elevated due to green "
             "energy investment. Debt levels stable year over year."},
    {"id": "DOC-EARN-2026-TCS-Q1", "ticker": "TCS.NS",
     "text": "TCS earnings call highlights steady deal wins in BFSI vertical, cautious "
             "commentary on discretionary IT spend in North America, margins held flat "
             "through cost optimization."},
    {"id": "DOC-EARN-2026-INFY-Q1", "ticker": "INFY.NS",
     "text": "Infosys management flagged softness in retail and manufacturing client "
             "budgets, offset by strong deal pipeline in AI-led transformation projects. "
             "Guidance maintained but tone cautious."},
    {"id": "DOC-SEBI-2026-HDFC", "ticker": "HDFCBANK.NS",
     "text": "HDFC Bank regulatory filing shows improving net interest margin post merger "
             "integration, asset quality stable, provisioning coverage ratio strengthened "
             "this quarter."},
    {"id": "DOC-EARN-2026-TATAMOTORS", "ticker": "TATAMOTORS.NS",
     "text": "Tata Motors reports strong JLR export volumes and improving EV segment "
             "traction domestically, though input cost inflation pressures margins "
             "modestly this quarter."},
]

POSITIVE_WORDS = {"strong", "growth", "improving", "strengthened", "steady", "maintained"}
NEGATIVE_WORDS = {"cautious", "softness", "pressure", "pressures", "elevated", "inflation"}

AGENT_ICONS = {"Momentum Agent": "📈", "Sentiment Agent": "📰", "Fundamentals (RAG) Agent": "📄"}


def synthetic_price_series(ticker, days=30):
    seed = int(hashlib.md5(ticker.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    price = 100 + rng.random() * 900
    prices, volumes = [], []
    for _ in range(days):
        price *= 1 + rng.uniform(-0.02, 0.02)
        prices.append(round(price, 2))
        volumes.append(int(rng.uniform(0.5e6, 3e6)))
    return prices, volumes


class DataFetcher:
    def __init__(self, force_degraded=False):
        self.force_degraded = force_degraded

    def get_price_history(self, ticker):
        if not self.force_degraded and YF_AVAILABLE:
            try:
                hist = yf.Ticker(ticker).history(period="1mo")
                if not hist.empty:
                    return hist["Close"].tolist(), hist["Volume"].tolist(), "live"
            except Exception:
                pass
        prices, volumes = synthetic_price_series(ticker)
        return prices, volumes, "synthetic (degraded fallback)"


# ---------------------------------------------------------------------------
# AGENTS
# ---------------------------------------------------------------------------

class MomentumAgent:
    name = "Momentum Agent"

    def run(self, ticker, fetcher):
        prices, volumes, status = fetcher.get_price_history(ticker)
        if len(prices) < 5:
            return {"score": 0, "confidence": 0.1, "reasoning": "Insufficient price history.",
                     "source": status, "prices": prices}
        momentum = (prices[-1] - prices[0]) / prices[0]
        avg_vol = statistics.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
        vol_z = (volumes[-1] - avg_vol) / (avg_vol + 1e-6)
        score = max(-1, min(1, momentum * 5))
        confidence = max(0.2, min(0.95, abs(momentum) * 8 + abs(vol_z) * 0.3))
        reasoning = (f"{len(prices)}-day momentum {momentum*100:.1f}%, "
                     f"volume vs avg {vol_z*100:+.0f}% ({status} data).")
        return {"score": round(score, 2), "confidence": round(confidence, 2),
                 "reasoning": reasoning, "source": status, "prices": prices}


class SentimentAgent:
    name = "Sentiment Agent"

    def run(self, ticker, fetcher):
        docs = [d for d in DOCUMENT_CORPUS if d["ticker"] == ticker]
        if not docs:
            return {"score": 0, "confidence": 0.2, "reasoning": "No sentiment source found.", "source": None}
        text = " ".join(d["text"].lower() for d in docs)
        pos = sum(text.count(w) for w in POSITIVE_WORDS)
        neg = sum(text.count(w) for w in NEGATIVE_WORDS)
        total = pos + neg
        score = 0 if total == 0 else (pos - neg) / total
        confidence = min(0.9, 0.3 + total * 0.1)
        reasoning = f"{pos} positive / {neg} negative signal words across {len(docs)} document(s)."
        return {"score": round(score, 2), "confidence": round(confidence, 2),
                 "reasoning": reasoning, "source": docs[0]["id"]}


class FundamentalsRAGAgent:
    name = "Fundamentals (RAG) Agent"

    def run(self, ticker, fetcher):
        docs = [d for d in DOCUMENT_CORPUS if d["ticker"] == ticker]
        if not docs:
            return {"score": 0, "confidence": 0.1,
                     "reasoning": "No matching filing/transcript retrieved.", "source": None}
        best = docs[0]
        text_lower = best["text"].lower()
        pos = sum(text_lower.count(w) for w in POSITIVE_WORDS)
        neg = sum(text_lower.count(w) for w in NEGATIVE_WORDS)
        score = 0 if (pos + neg) == 0 else (pos - neg) / (pos + neg)
        reasoning = f"Retrieved [{best['id']}]: \"{best['text'][:110]}...\""
        return {"score": round(score, 2), "confidence": 0.75, "reasoning": reasoning, "source": best["id"]}


# ---------------------------------------------------------------------------
# SYNTHESIS LAYER
# ---------------------------------------------------------------------------

RISK_WEIGHTS = {
    "Conservative": {"Momentum Agent": 0.15, "Sentiment Agent": 0.35, "Fundamentals (RAG) Agent": 0.50},
    "Moderate":     {"Momentum Agent": 0.34, "Sentiment Agent": 0.33, "Fundamentals (RAG) Agent": 0.33},
    "Aggressive":   {"Momentum Agent": 0.55, "Sentiment Agent": 0.25, "Fundamentals (RAG) Agent": 0.20},
}


def synthesize(results, risk_profile):
    weights = RISK_WEIGHTS[risk_profile]
    weighted_score = sum(results[a]["score"] * weights[a] for a in results)
    avg_confidence = sum(results[a]["confidence"] * weights[a] for a in results)
    scores = [results[a]["score"] for a in results]
    conflicting = max(scores) > 0.15 and min(scores) < -0.15

    if conflicting:
        label = "HOLD (conflicting signals)"
        avg_confidence *= 0.6
    elif weighted_score > 0.2:
        label = "BUY"
    elif weighted_score < -0.2:
        label = "SELL"
    else:
        label = "HOLD"

    citations = [results[a]["source"] for a in results if results[a]["source"]]
    return {"label": label, "score": round(weighted_score, 2),
            "confidence": round(min(0.95, max(0.05, avg_confidence)), 2),
            "conflicting": conflicting, "citations": citations}


# ---------------------------------------------------------------------------
# PERFORMANCE LOGGER
# ---------------------------------------------------------------------------

LOG_FILE = "session_log.csv"


class PerformanceLogger:
    def __init__(self):
        self.session_count = 0
        self.latencies = []
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "ticker", "risk_profile", "latency_sec",
                                         "recommendation", "confidence", "risk_concentration"])

    def log(self, ticker, risk_profile, latency, recommendation, confidence, concentration):
        self.session_count += 1
        self.latencies.append(latency)
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), ticker, risk_profile,
                                     f"{latency:.3f}", recommendation, confidence, f"{concentration:.2f}"])

    def summary(self):
        avg_latency = statistics.mean(self.latencies) if self.latencies else 0
        return f"Analyses run: {self.session_count}   |   Avg latency: {avg_latency:.2f}s"


def risk_concentration(watchlist):
    if not watchlist:
        return 0.0
    sectors = [SECTOR_MAP.get(t, "Other") for t in watchlist]
    top_sector_count = max(sectors.count(s) for s in set(sectors))
    return top_sector_count / len(watchlist)


def score_color(score):
    if score > 0.15:
        return "#22c55e"
    if score < -0.15:
        return "#ef4444"
    return "#eab308"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

TICKERS = list(SECTOR_MAP.keys())
BG = "#0f1117"
CARD = "#1a1d27"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Agent Financial Intelligence System - PS-01")
        self.geometry("1250x820")
        self.configure(fg_color=BG)

        self.fetcher = DataFetcher()
        self.logger = PerformanceLogger()
        self.watchlist = []
        self.agents = [MomentumAgent(), SentimentAgent(), FundamentalsRAGAgent()]
        self._dot_count = 0
        self._analyzing = False

        self._build_ui()

    # ---------------- UI BUILD ----------------

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color=CARD, corner_radius=0, height=90)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="⚡ Multi-Agent Financial Intelligence System",
                     font=ctk.CTkFont(size=24, weight="bold"), text_color="#e5e7eb").pack(pady=(16, 0))
        ctk.CTkLabel(header, text="PS-01  ·  Explainable, personalized investment intelligence",
                     font=ctk.CTkFont(size=12), text_color="#9ca3af").pack(pady=(0, 14))

        controls = ctk.CTkFrame(self, fg_color=CARD, corner_radius=12)
        controls.pack(fill="x", padx=18, pady=12)

        self.ticker_var = ctk.StringVar(value=TICKERS[0])
        ctk.CTkLabel(controls, text="Ticker").grid(row=0, column=0, padx=(16, 4), pady=14)
        ctk.CTkOptionMenu(controls, values=TICKERS, variable=self.ticker_var, width=140).grid(row=0, column=1, padx=6)

        self.risk_var = ctk.StringVar(value="Moderate")
        ctk.CTkLabel(controls, text="Risk profile").grid(row=0, column=2, padx=(16, 4))
        ctk.CTkOptionMenu(controls, values=list(RISK_WEIGHTS.keys()), variable=self.risk_var, width=140).grid(row=0, column=3, padx=6)

        self.degraded_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(controls, text="Simulate degraded feed", variable=self.degraded_var).grid(row=0, column=4, padx=18)

        ctk.CTkButton(controls, text="＋ Watchlist", command=self.add_to_watchlist,
                      fg_color="#374151", hover_color="#4b5563", width=110).grid(row=0, column=5, padx=6)

        self.analyze_btn = ctk.CTkButton(controls, text="▶  Analyze", command=self.run_analysis,
                                          fg_color="#16a34a", hover_color="#15803d",
                                          font=ctk.CTkFont(weight="bold"), width=140, height=36)
        self.analyze_btn.grid(row=0, column=6, padx=16, pady=10)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=18, pady=(0, 10))

        # ---- LEFT: agent cards ----
        left = ctk.CTkFrame(body, fg_color=CARD, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ctk.CTkLabel(left, text="Live Agent Signals", font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(14, 6))

        self.agent_widgets = {}
        for agent in self.agents:
            card = ctk.CTkFrame(left, fg_color="#20242f", corner_radius=10)
            card.pack(fill="x", padx=14, pady=6)
            top = ctk.CTkFrame(card, fg_color="transparent")
            top.pack(fill="x", padx=10, pady=(10, 2))
            ctk.CTkLabel(top, text=f"{AGENT_ICONS.get(agent.name,'')}  {agent.name}",
                         font=ctk.CTkFont(weight="bold")).pack(side="left")
            conf_lbl = ctk.CTkLabel(top, text="--", text_color="#9ca3af")
            conf_lbl.pack(side="right")

            score_bar = ctk.CTkProgressBar(card, height=10, progress_color="#6b7280")
            score_bar.pack(fill="x", padx=10, pady=(4, 2))
            score_bar.set(0.5)

            reasoning_lbl = ctk.CTkLabel(card, text="Waiting for analysis...", justify="left",
                                          wraplength=330, text_color="#9ca3af", font=ctk.CTkFont(size=11))
            reasoning_lbl.pack(anchor="w", padx=10, pady=(2, 10))

            self.agent_widgets[agent.name] = {"bar": score_bar, "conf": conf_lbl, "reason": reasoning_lbl}

        # ---- MIDDLE: recommendation ----
        mid = ctk.CTkFrame(body, fg_color=CARD, corner_radius=12)
        mid.pack(side="left", fill="both", expand=True, padx=8)
        ctk.CTkLabel(mid, text="Synthesized Recommendation", font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(14, 6))

        self.rec_panel = ctk.CTkFrame(mid, fg_color="#20242f", corner_radius=10)
        self.rec_panel.pack(fill="x", padx=14, pady=6)
        self.rec_label = ctk.CTkLabel(self.rec_panel, text="—", font=ctk.CTkFont(size=32, weight="bold"))
        self.rec_label.pack(pady=(16, 4))
        self.conf_bar = ctk.CTkProgressBar(self.rec_panel, height=14, progress_color="#3b82f6")
        self.conf_bar.pack(fill="x", padx=20, pady=(0, 4))
        self.conf_bar.set(0)
        self.conf_text = ctk.CTkLabel(self.rec_panel, text="Confidence: --", text_color="#9ca3af")
        self.conf_text.pack(pady=(0, 6))
        self.rec_detail = ctk.CTkLabel(self.rec_panel, text="Run an analysis to see output.",
                                        wraplength=340, justify="left", font=ctk.CTkFont(size=11))
        self.rec_detail.pack(pady=(0, 16), padx=16)

        ctk.CTkLabel(mid, text="Price Trend", font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(10, 4))
        self.chart_frame = ctk.CTkFrame(mid, fg_color="#20242f", corner_radius=10)
        self.chart_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._build_chart()

        # ---- RIGHT: tabs ----
        right = ctk.CTkFrame(body, fg_color=CARD, corner_radius=12)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        tabs = ctk.CTkTabview(right, fg_color="#20242f", segmented_button_selected_color="#3b82f6")
        tabs.pack(fill="both", expand=True, padx=12, pady=14)
        tabs.add("Watchlist")
        tabs.add("Performance")

        self.watchlist_box = ctk.CTkTextbox(tabs.tab("Watchlist"), fg_color="#0f1117")
        self.watchlist_box.pack(fill="both", expand=True, padx=4, pady=4)

        self.perf_label = ctk.CTkLabel(tabs.tab("Performance"), text="No analyses yet.",
                                        justify="left", anchor="nw")
        self.perf_label.pack(fill="both", expand=True, padx=8, pady=8)

        # ---- BOTTOM: reasoning trace ----
        bottom = ctk.CTkFrame(self, fg_color=CARD, corner_radius=12)
        bottom.pack(fill="both", expand=False, padx=18, pady=(0, 16))
        ctk.CTkLabel(bottom, text="🧠  Agent Reasoning Trace", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=14, pady=(10, 0))
        self.trace_box = ctk.CTkTextbox(bottom, height=130, fg_color="#0f1117")
        self.trace_box.pack(fill="both", expand=True, padx=14, pady=12)

    def _build_chart(self):
        self.fig = Figure(figsize=(3.6, 2.4), dpi=100)
        self.fig.patch.set_facecolor("#20242f")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#20242f")
        for spine in self.ax.spines.values():
            spine.set_color("#3f4451")
        self.ax.tick_params(colors="#9ca3af", labelsize=7)
        self.ax.set_title("No data yet", color="#9ca3af", fontsize=9)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def _update_chart(self, ticker, prices):
        self.ax.clear()
        self.ax.set_facecolor("#20242f")
        color = "#22c55e" if prices[-1] >= prices[0] else "#ef4444"
        self.ax.plot(prices, color=color, linewidth=2)
        self.ax.fill_between(range(len(prices)), prices, min(prices), color=color, alpha=0.12)
        self.ax.set_title(f"{ticker}  (last {len(prices)} sessions)", color="#e5e7eb", fontsize=9)
        self.ax.tick_params(colors="#9ca3af", labelsize=7)
        for spine in self.ax.spines.values():
            spine.set_color("#3f4451")
        self.canvas.draw()

    # ---------------- ACTIONS ----------------

    def add_to_watchlist(self):
        t = self.ticker_var.get()
        if t not in self.watchlist:
            self.watchlist.append(t)
        self.watchlist_box.delete("1.0", "end")
        for w in self.watchlist:
            self.watchlist_box.insert("end", f"●  {w}   ({SECTOR_MAP.get(w, 'Other')})\n\n")

    def run_analysis(self):
        self._analyzing = True
        self.analyze_btn.configure(state="disabled", fg_color="#374151")
        self._animate_button()
        threading.Thread(target=self._analyze_worker, daemon=True).start()

    def _animate_button(self):
        if not self._analyzing:
            return
        self._dot_count = (self._dot_count + 1) % 4
        self.analyze_btn.configure(text="Analyzing" + "." * self._dot_count)
        self.after(300, self._animate_button)

    def _analyze_worker(self):
        ticker = self.ticker_var.get()
        risk_profile = self.risk_var.get()
        self.fetcher.force_degraded = self.degraded_var.get()

        start = time.perf_counter()
        results = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {ex.submit(agent.run, ticker, self.fetcher): agent.name for agent in self.agents}
            for fut, name in futures.items():
                results[name] = fut.result()
        latency = time.perf_counter() - start

        synthesis = synthesize(results, risk_profile)
        concentration = risk_concentration(self.watchlist or [ticker])
        self.logger.log(ticker, risk_profile, latency, synthesis["label"], synthesis["confidence"], concentration)

        self.after(0, self._update_ui, ticker, results, synthesis, latency, concentration)

    def _update_ui(self, ticker, results, synthesis, latency, concentration):
        self._analyzing = False
        for name, res in results.items():
            w = self.agent_widgets[name]
            norm = (res["score"] + 1) / 2
            w["bar"].configure(progress_color=score_color(res["score"]))
            w["bar"].set(norm)
            w["conf"].configure(text=f"{res['confidence']:.0%} conf")
            w["reason"].configure(text=res["reasoning"])

        color = score_color(synthesis["score"])
        if "HOLD (conflicting" in synthesis["label"]:
            color = "#eab308"
        self.rec_label.configure(text=synthesis["label"], text_color=color)
        self.conf_bar.configure(progress_color=color)
        self.conf_bar.set(synthesis["confidence"])
        self.conf_text.configure(text=f"Confidence: {synthesis['confidence']:.0%}")

        cite_str = ", ".join(synthesis["citations"]) if synthesis["citations"] else "no citation available"
        conflict_note = "⚠ Conflicting agent signals detected.\n" if synthesis["conflicting"] else ""
        self.rec_detail.configure(text=f"{conflict_note}Weighted score: {synthesis['score']:+.2f}\nSources: {cite_str}")

        self.perf_label.configure(
            text=f"{self.logger.summary()}\n\nRisk concentration: {concentration:.0%}\n"
                 f"Last run latency: {latency:.2f}s\nLast recommendation: {synthesis['label']}")

        if "prices" in results.get("Momentum Agent", {}):
            self._update_chart(ticker, results["Momentum Agent"]["prices"])

        self.trace_box.insert("end", f"\n--- {datetime.now().strftime('%H:%M:%S')}  |  {ticker}  |  {self.risk_var.get()} ---\n")
        for name, res in results.items():
            self.trace_box.insert("end", f"[{name}] {res['reasoning']} (source: {res['source']})\n")
        self.trace_box.insert("end", f"[Synthesis] -> {synthesis['label']} (conf {synthesis['confidence']:.0%}, latency {latency:.2f}s)\n")
        self.trace_box.see("end")

        self.analyze_btn.configure(state="normal", text="▶  Analyze", fg_color="#16a34a")


if __name__ == "__main__":
    app = App()
    app.mainloop()