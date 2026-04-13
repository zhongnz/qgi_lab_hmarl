#!/usr/bin/env python3
"""Build a PDF preview for the Google Slides import deck."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "docs" / "reports"
RUNS = ROOT / "runs"

OUT = REPORTS / "2026-03-31_hmarl_project_update_preview.pdf"
V3_RUN = RUNS / "local_full_train_2026-03-11_transit_rebalanced_v3"
EVAL_TRACE_CSV = V3_RUN / "eval_trace.csv"
ROLE_DIAG_IMG = REPORTS / "2026-03-31_role_reward_diagnostics.png"

W, H = 1600, 900

BG = (244, 239, 230)
PAPER = (255, 250, 242)
INK = (20, 35, 54)
MUTED = (84, 98, 117)
NAVY = (20, 56, 92)
NAVY_2 = (34, 73, 110)
TEAL = (43, 122, 120)
TEAL_SOFT = (235, 247, 245)
ORANGE_SOFT = (253, 243, 232)
SAND = (231, 216, 189)
LINE = (221, 209, 188)
WHITE = (255, 255, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_H1 = _font(42, True)
FONT_H2 = _font(28, True)
FONT_H3 = _font(22, True)
FONT_BODY = _font(20, False)
FONT_BODY_SM = _font(16, False)
FONT_BODY_XS = _font(14, False)
FONT_LABEL = _font(12, True)
FONT_PAGE = _font(18, True)
FONT_MONO = _font(16, False)


def _read_v3_metrics() -> dict[str, float]:
    data = json.loads((V3_RUN / "eval_result.json").read_text())["mean"]
    return {
        "total_reward": float(data["total_reward"]),
        "on_time_rate": float(data["on_time_rate"]),
        "completed_arrivals": float(data["completed_arrivals"]),
        "total_vessels_served": float(data["total_vessels_served"]),
        "dock_utilization": float(data["dock_utilization"]),
        "total_ops_cost_usd": float(data["total_ops_cost_usd"]),
    }


def _ensure_role_reward_plot() -> Path:
    if ROLE_DIAG_IMG.exists() and ROLE_DIAG_IMG.stat().st_mtime >= EVAL_TRACE_CSV.stat().st_mtime:
        return ROLE_DIAG_IMG

    with EVAL_TRACE_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    t = [float(row["t"]) for row in rows]
    series = [
        ("Vessel reward", [float(row["avg_vessel_reward"]) for row in rows], (43 / 255, 122 / 255, 120 / 255)),
        ("Port reward", [float(row["avg_port_reward"]) for row in rows], (214 / 255, 126 / 255, 63 / 255)),
        ("Coordinator reward", [float(row["coordinator_reward"]) for row in rows], (105 / 255, 110 / 255, 196 / 255)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), dpi=180)
    fig.patch.set_facecolor((250 / 255, 248 / 255, 244 / 255))
    for ax, (title, values, color) in zip(axes, series):
        ax.plot(t, values, color=color, linewidth=2.2)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("reward", fontsize=10)
    fig.suptitle("Per-step reward diagnostics by role", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(ROLE_DIAG_IMG, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return ROLE_DIAG_IMG


def _new_slide() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), PAPER)
    return img, ImageDraw.Draw(img)


def _rr(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, outline=LINE, radius=22) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 0)


def _text(draw, xy, text, font, fill=INK, max_width=None, line_spacing=6, anchor=None):
    x, y = xy
    if max_width is None:
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
        bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=line_spacing, anchor=anchor)
        return bbox[3] - bbox[1]
    wrapped = _wrap_text(draw, text, font, max_width)
    draw.multiline_text((x, y), wrapped, font=font, fill=fill, spacing=line_spacing, anchor=anchor)
    bbox = draw.multiline_textbbox((x, y), wrapped, font=font, spacing=line_spacing, anchor=anchor)
    return bbox[3] - bbox[1]


def _wrap_text(draw, text, font, max_width):
    words = text.split()
    if not words:
        return text
    lines = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


def _bullets(draw, x, y, width, items, font=FONT_BODY, color=INK, gap=14):
    cy = y
    for item in items:
        draw.ellipse((x, cy + 10, x + 8, cy + 18), fill=color)
        h = _text(draw, (x + 20, cy), item, font, color, max_width=width - 24)
        cy += h + gap
    return cy


def _top_band(draw, title, eyebrow, page):
    draw.rectangle((0, 0, W, 88), fill=NAVY)
    _text(draw, (54, 20), eyebrow.upper(), FONT_LABEL, SAND)
    _text(draw, (54, 42), title, FONT_H2, PAPER)
    _text(draw, (1535, 22), page, FONT_PAGE, SAND, anchor="ra")


def _paste_image(base: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    if not path.exists():
        return
    img = Image.open(path).convert("RGB")
    fitted = ImageOps.contain(img, (box[2] - box[0], box[3] - box[1]))
    bg = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), (245, 242, 236))
    px = (bg.width - fitted.width) // 2
    py = (bg.height - fitted.height) // 2
    bg.paste(fitted, (px, py))
    base.paste(bg, (box[0], box[1]))


def slide_1():
    img, d = _new_slide()
    img.paste(Image.new("RGB", (W, H), BG))
    _rr(d, (1040, 90, 1450, 310), (229, 238, 245), None)
    _rr(d, (1100, 395, 1400, 555), ORANGE_SOFT, None)
    _text(d, (84, 102), "Hierarchical Multi-Agent Reinforcement Learning for Congestion-Aware Maritime Scheduling", FONT_H1, INK, max_width=930)
    _text(d, (88, 300), "Project update: simulator background, reward redesign, continuous + ground_truth final run path, current progress, and next steps.", FONT_BODY, MUTED, max_width=860)
    _rr(d, (84, 410, 1020, 560), NAVY, None)
    _text(d, (112, 450), "We now have a clean final path for the project: continuous scheduling, reliable forecasts, strong diagnostics, and a reproducible local run plan.", FONT_BODY, PAPER, max_width=870)
    _text(d, (1075, 122), "SPRING 2026", FONT_LABEL, MUTED)
    _text(d, (1075, 156), "Independent study", FONT_H3, NAVY)
    _text(d, (1075, 212), "Supervisor:\nProf. Amine Aboussalah", FONT_BODY_SM, INK, max_width=300)
    _text(d, (1130, 418), "MARCH 31 UPDATE", FONT_LABEL, MUTED)
    _text(d, (1130, 452), "Final path:\ncontinuous + GT", FONT_BODY_SM, NAVY, max_width=210)
    cards = [
        (84, 635, 465, 800, "RESEARCH FOCUS", "HMARL", "Coordinator, vessel, and port agents"),
        (565, 635, 995, 800, "CURRENT SETUP", "Cont. + GT", "Continuous environment with reliable forecasts"),
        (1015, 635, 1420, 800, "FORMAT", "12 slides", "Presentation-ready update deck"),
    ]
    for x1, y1, x2, y2, label, value, note in cards:
        _rr(d, (x1, y1, x2, y2), WHITE)
        _text(d, (x1 + 24, y1 + 18), label, FONT_LABEL, MUTED)
        _text(d, (x1 + 24, y1 + 54), value, FONT_H2, NAVY)
        _text(d, (x1 + 24, y1 + 104), note, FONT_BODY_SM, MUTED, max_width=(x2 - x1 - 48))
    return img


def slide_2():
    img, d = _new_slide()
    _top_band(d, "Why this project matters", "Context", "02")
    _rr(d, (78, 135, 820, 780), (238, 246, 251))
    _text(d, (110, 170), "Operational context", FONT_H3, NAVY)
    _bullets(d, 116, 230, 650, [
        "Ports have limited berth capacity and nonzero service time.",
        "Vessels trade off speed, fuel burn, emissions, and lateness.",
        "Poor local choices can create downstream congestion and idle capacity elsewhere.",
        "The system is naturally multi-agent, asynchronous, and resource-constrained.",
    ], font=FONT_BODY)
    boxes = [
        (870, 135, 1185, 405, "Congestion", "Too many arrivals to one port can create queueing and delay at the whole network level."),
        (1210, 135, 1525, 405, "Cost", "Faster sailing can improve timing but increases fuel use, emissions, and operating cost."),
        (870, 435, 1185, 705, "Coordination", "Ports and vessels need global guidance, not only local reactive behavior."),
        (1210, 435, 1525, 705, "Research value", "This is a strong testbed for hierarchical coordination under delayed information."),
    ]
    for x1, y1, x2, y2, title, text in boxes:
        _rr(d, (x1, y1, x2, y2), WHITE)
        _text(d, (x1 + 22, y1 + 24), title, FONT_H3, NAVY)
        _text(d, (x1 + 22, y1 + 84), text, FONT_BODY_SM, INK, max_width=x2 - x1 - 44)
    return img


def slide_3():
    img, d = _new_slide()
    _top_band(d, "What we are trying to learn", "Questions", "03")
    _rr(d, (78, 138, 1522, 250), NAVY, None)
    _text(d, (110, 168), "Umbrella question", FONT_H3, SAND)
    _text(d, (110, 206), "Can hierarchical MARL improve maritime scheduling under congestion and resource constraints?", FONT_BODY, PAPER, max_width=1330)
    questions = [
        ("RQ1", "How can heterogeneous agents coordinate using shared congestion forecasts?"),
        ("RQ2", "Does proactive coordination improve over independent and reactive baselines?"),
        ("RQ3", "Which forecast horizons and sharing strategies maximize decision quality?"),
        ("RQ4", "How do coordination improvements affect fuel, delay, and carbon cost?"),
    ]
    positions = [(78, 300), (810, 300), (78, 520), (810, 520)]
    for (rq, txt), (x, y) in zip(questions, positions):
        _rr(d, (x, y, x + 670, y + 170), WHITE)
        _text(d, (x + 24, y + 20), rq, FONT_H3, NAVY)
        _text(d, (x + 24, y + 72), txt, FONT_BODY_SM, INK, max_width=620)
    _text(d, (800, 815), "The project is organized around coordination, baselines, forecast design, and economics.", FONT_BODY_SM, MUTED, anchor="ma")
    return img


def slide_4():
    img, d = _new_slide()
    _top_band(d, "Current simulator setup", "Environment", "04")
    cards = [
        ("Topology", "5 ports, 8 vessels, 3 docks per port"),
        ("Time scale", "1 hour per step and 6 hour default service time"),
        ("Travel", "Synthetic port-distance matrix with routes from 22 to 98 nautical miles"),
        ("Fuel", "Default fuel is 100, with refueling only after real service completion"),
    ]
    y = 142
    for title, text in cards:
        _rr(d, (78, y, 540, y + 125), WHITE)
        _text(d, (102, y + 18), title, FONT_H3, NAVY)
        _text(d, (102, y + 62), text, FONT_BODY_SM, INK, max_width=390)
        y += 140
    _rr(d, (570, 142, 1522, 780), (250, 248, 244))
    _text(d, (600, 170), "Synthetic port distance matrix (nautical miles)", FONT_H3, NAVY)
    _text(d, (600, 214), "Rows and columns are ports P0–P4; the diagonal is 0, and the matrix is symmetric.", FONT_BODY_SM, MUTED, max_width=860)
    _text(d, (600, 248), "The values are synthetic and scaled so multiple voyages can complete within one rollout.", FONT_BODY_SM, MUTED, max_width=860)
    labels = ["", "P0", "P1", "P2", "P3", "P4"]
    matrix = [
        ["P0", "0", "84", "22", "34", "78"],
        ["P1", "84", "0", "98", "61", "54"],
        ["P2", "22", "98", "0", "55", "59"],
        ["P3", "34", "61", "55", "0", "82"],
        ["P4", "78", "54", "59", "82", "0"],
    ]
    sx, sy, cw, ch = 625, 330, 145, 60
    for c, label in enumerate(labels):
        _rr(d, (sx + c * cw, sy, sx + (c + 1) * cw - 8, sy + ch - 8), (228, 236, 244))
        _text(d, (sx + c * cw + (cw - 8) / 2, sy + 17), label, FONT_BODY_SM, NAVY, anchor="ma")
    for r, row in enumerate(matrix, start=1):
        for c, val in enumerate(row):
            fill = WHITE if c else (247, 244, 239)
            _rr(d, (sx + c * cw, sy + r * ch, sx + (c + 1) * cw - 8, sy + (r + 1) * ch - 8), fill)
            _text(d, (sx + c * cw + (cw - 8) / 2, sy + r * ch + 17), val, FONT_BODY_SM, NAVY if c == 0 else INK, anchor="ma")
    return img


def slide_5():
    img, d = _new_slide()
    _top_band(d, "How the current baseline is trained and evaluated", "Experiment", "05")
    _rr(d, (78, 145, 760, 780), (245, 250, 252))
    _text(d, (106, 175), "Main operating baseline", FONT_H3, NAVY)
    _bullets(d, 112, 235, 610, [
        "Algorithm: MAPPO with centralized training and decentralized execution",
        "Parameter sharing across vessels and across ports",
        "100 training iterations and rollout_length = 64",
        "Environment horizon = 69",
        "Seed = 42",
        "Local CPU run for the reported baseline",
    ], font=FONT_BODY_SM)
    _rr(d, (795, 145, 1522, 420), ORANGE_SOFT)
    _text(d, (823, 175), "Final completion path", FONT_H3, NAVY)
    _bullets(d, 829, 235, 620, [
        "continuous episodes",
        "ground_truth forecasts",
        "one artifact run plus one five-seed run",
        "designed to isolate control quality while finishing the project",
    ], font=FONT_BODY_SM)
    _rr(d, (795, 460, 1522, 780), TEAL_SOFT)
    _text(d, (823, 490), "Primary evaluation metrics", FONT_H3, NAVY)
    _bullets(d, 829, 550, 620, [
        "On-time rate",
        "Completed arrivals",
        "Port service events",
        "Dock utilization",
        "Total operating cost",
    ], font=FONT_BODY_SM)
    return img


def slide_6():
    img, d = _new_slide()
    _top_band(d, "Three decision layers with delayed coordination", "Architecture", "06")
    items = [
        ("Fleet coordinator", "Strategic destination guidance\nSlowest decision cadence", (245, 248, 252)),
        ("Vessel agents", "Speed control\nArrival-slot requests", WHITE),
        ("Port agents", "Request acceptance\nService and berth decisions", ORANGE_SOFT),
    ]
    xs = [110, 560, 1010]
    for (title, text, fill), x in zip(items, xs):
        _rr(d, (x, 275, x + 360, 515), fill)
        _text(d, (x + 26, 305), title, FONT_H3, NAVY)
        _text(d, (x + 26, 380), text, FONT_BODY, INK, max_width=300)
    _text(d, (502, 378), "→", _font(48, True), (214, 126, 63))
    _text(d, (952, 378), "→", _font(48, True), (214, 126, 63))
    _rr(d, (110, 620, 1490, 735), ORANGE_SOFT, None)
    _text(d, (150, 655), "The environment is intentionally asynchronous, so the policies must coordinate under delayed information and physical travel time.", FONT_BODY, INK, max_width=1300)
    return img


def slide_7():
    img, d = _new_slide()
    _top_band(d, "How one simulation step works", "Mechanics", "07")
    _rr(d, (78, 145, 1040, 780), (245, 250, 252))
    _text(d, (106, 175), "Transition kernel", FONT_H3, NAVY)
    steps = [
        ("1. Deliver messages", "Apply due actions and delayed communications."),
        ("2. Advance vessels", "Update motion, fuel burn, emissions, and arrivals."),
        ("3. Process ports", "Update queue, berth service, service completion, and refueling."),
        ("4. Compute rewards", "Log diagnostics and compute role-specific rewards."),
    ]
    y = 240
    for title, text in steps:
        _rr(d, (106, y, 995, y + 86), WHITE)
        _text(d, (128, y + 12), title, FONT_BODY_SM, NAVY)
        _text(d, (128, y + 42), text, FONT_BODY_SM, INK, max_width=820)
        y += 100
    _rr(d, (1080, 145, 1522, 780), ORANGE_SOFT)
    _text(d, (1108, 175), "Why this matters", FONT_H3, NAVY)
    _bullets(d, 1114, 245, 345, [
        "The simulator evolves through communication, travel, berth service, and reward calculation in sequence.",
        "That is why arrivals, delays, and service completions appear at different times in the logs.",
        "The next two slides separate reward intuition from reward formulas so the audience can follow both.",
    ], font=FONT_BODY_SM)
    return img


def slide_8():
    img, d = _new_slide()
    _top_band(d, "Reward structure in plain language", "Rewards", "08")
    cards = [
        ("Vessel reward", "Reward = arrival and punctuality minus travel cost", ["Arrival", "On-time arrival"], ["Fuel used", "Delay added", "CO2 emitted", "Transit time", "Schedule delay"], WHITE, 180),
        ("Port reward", "Reward = responsive berth management minus congestion cost", ["Accepted requests", "Vessels served"], ["Rejected requests", "Queue waiting", "Idle docks"], TEAL_SOFT, 180),
        ("Coordinator reward", "Reward = system flow and utilization minus network-wide cost", ["Accepted requests", "Vessels served", "Berth utilization"], ["Fuel", "Queue", "Idle docks", "Delay", "Schedule delay", "CO2", "Rejected requests"], WHITE, 200),
    ]
    y = 130
    for title, headline, rewards, penalties, fill, h in cards:
        _rr(d, (78, y, 1522, y + h), fill)
        _text(d, (106, y + 18), title, FONT_H3, NAVY)
        _text(d, (106, y + 60), headline, FONT_BODY_SM, INK, max_width=1310)
        _rr(d, (106, y + 92, 680, y + h - 16), (242, 248, 244))
        _rr(d, (710, y + 92, 1492, y + h - 16), (252, 245, 239))
        _text(d, (128, y + 110), "What it rewards", FONT_LABEL, NAVY)
        _text(d, (732, y + 110), "What it penalizes", FONT_LABEL, NAVY)
        _text(d, (128, y + 142), ", ".join(rewards), FONT_BODY_XS, INK, max_width=500)
        _text(d, (732, y + 142), ", ".join(penalties), FONT_BODY_XS, INK, max_width=700)
        y += h + 18
    _text(d, (800, 860), "This is the audience-facing summary. The next slide shows the actual symbolic formulas used in the simulator.", FONT_BODY_SM, MUTED, anchor="ma")
    return img


def slide_9():
    img, d = _new_slide()
    _top_band(d, "Actual reward formulas used in the simulator", "Rewards", "09")
    _rr(d, (78, 145, 1522, 780), ORANGE_SOFT)
    _text(d, (106, 175), "Symbolic form", FONT_H3, NAVY)
    formulas = [
        ("Vessel reward", "r_V(t) = r_arr·1[arrived] + r_on·1[on_time]\n          − (w_fΔfuel + w_dΔdelay + w_eΔCO2 + w_tΔtransit + w_sΔsched)", WHITE, 220, 160),
        ("Port reward", "r_P(t) = r_acc·accepted + r_srv·served\n          − r_rej·rejected − (queue·dt + w_idle·idle_docks)", TEAL_SOFT, 400, 136),
        ("Coordinator reward", "r_C(t) = r_acc·accepted + r_srv·served + r_util·avg_occupied\n          − (w_fΔfuel + w_qavg_queue + w_iavg_idle + w_dΔdelay + w_sΔsched + w_eΔCO2 + r_rej·rejected)", WHITE, 556, 176),
    ]
    for title, formula, fill, y, h in formulas:
        _rr(d, (106, y, 1492, y + h), fill)
        _text(d, (128, y + 18), title, FONT_H3, NAVY)
        _text(d, (128, y + 76), formula, FONT_MONO, INK, max_width=1310)
    _text(d, (800, 840), "Δ terms are per-step increments; accepted, rejected, and served are event counts for the current step.", FONT_BODY_SM, MUTED, anchor="ma")
    return img


def slide_10():
    img, d = _new_slide()
    _top_band(d, "What changed since March 6", "Progress", "10")
    items = [
        ("Corrected vessel behavior", "No more collapse to constant minimum speed.\nRequested arrival times now become real schedule deadlines.", TEAL_SOFT),
        ("Fuel and stalling", "Fuel exhaustion now causes true mid-route stalling.\nDepartures are checked for fuel feasibility.", WHITE),
        ("Port service realism", "Ports track actual vessel IDs.\nRefueling happens after real service completion.", ORANGE_SOFT),
        ("Reward redesign", "Rewards moved from mostly aggregate penalties to event-driven, role-specific signals.", WHITE),
        ("Final run cleanup", "Moved the finish line to continuous + ground_truth and fixed training resets to vary reproducibly.", TEAL_SOFT),
        ("Visibility", "Action logs, event logs, reward components, and confidence diagnostics were added.", WHITE),
    ]
    coords = [(78, 150), (542, 150), (1006, 150), (78, 470), (542, 470), (1006, 470)]
    for (title, text, fill), (x, y) in zip(items, coords):
        _rr(d, (x, y, x + 438, y + 220), fill)
        _text(d, (x + 22, y + 20), title, FONT_H3, NAVY)
        _text(d, (x + 22, y + 86), text, FONT_BODY_SM, INK, max_width=390)
    return img


def slide_11():
    img, d = _new_slide()
    _top_band(d, "Why the current results are more trustworthy", "Diagnostics", "11")
    _rr(d, (78, 145, 560, 780), WHITE)
    _text(d, (106, 175), "Transparency upgrades", FONT_H3, NAVY)
    _bullets(d, 112, 235, 395, [
        "Grouped plots separate aggregate, vessel, port, and coordinator behavior.",
        "Per-step eval trace, action log, and event log are available.",
        "Reward-component decomposition shows which terms dominate.",
        "Policy-confidence metrics make near-uniform coordinator behavior easier to diagnose.",
    ], font=FONT_BODY_SM)
    _rr(d, (590, 145, 1522, 780), (250, 248, 244))
    _paste_image(img, _ensure_role_reward_plot(), (640, 190, 1460, 520))
    _text(d, (680, 548), "Role-specific reward traces over one evaluation episode. We show all three because vessel, port, and coordinator each optimize a different part of the task.", FONT_BODY_SM, MUTED, max_width=760)
    chips = [
        (670, "TRACE FILE", "eval_trace.csv"),
        (900, "ACTION LOG", "eval_action_trace.csv"),
        (1175, "EVENT LOG", "eval_event_log.csv"),
    ]
    for x, label, value in chips:
        _rr(d, (x, 610, x + 190, 700), TEAL_SOFT)
        _text(d, (x + 14, 628), label, FONT_LABEL, MUTED)
        _text(d, (x + 14, 662), value, FONT_BODY_SM, NAVY, max_width=160)
    return img


def slide_12(v3):
    img, d = _new_slide()
    _top_band(d, "Current recommended baseline: transit-rebalanced v3", "Baseline", "12")
    _rr(d, (78, 145, 620, 780), NAVY, None)
    _text(d, (110, 178), "Preferred operating balance", FONT_H3, PAPER)
    metrics = [
        ("TOTAL REWARD", f"{v3['total_reward']:.2f}"),
        ("ON-TIME RATE", f"{v3['on_time_rate']:.3f}"),
        ("COMPLETED ARRIVALS", f"{v3['completed_arrivals']:.1f}"),
        ("PORT SERVICE EVENTS", f"{v3['total_vessels_served']:.1f}"),
        ("DOCK UTILIZATION", f"{v3['dock_utilization']:.2f}"),
        ("OPS COST", f"${v3['total_ops_cost_usd']/1_000_000:.3f}M"),
    ]
    coords = [(110, 255), (315, 255), (110, 395), (315, 395), (110, 535), (315, 535)]
    for (label, value), (x, y) in zip(metrics, coords):
        _rr(d, (x, y, x + 175, y + 105), NAVY_2, None)
        _text(d, (x + 12, y + 12), label, FONT_LABEL, SAND)
        _text(d, (x + 12, y + 46), value, FONT_H3, PAPER)
    _rr(d, (650, 145, 1522, 780), (250, 248, 244))
    _paste_image(img, V3_RUN / "training_curves.png", (670, 168, 1500, 757))
    return img


def slide_13():
    img, d = _new_slide()
    _top_band(d, "Final full-scale run plan", "Run plan", "13")
    panels = [
        ("Artifact run", 78, (245, 250, 252), [
            ("ENVIRONMENT", "Continuous"),
            ("FORECAST", "Ground truth"),
            ("ITERATIONS", "100"),
            ("SEED", "42"),
            ("PURPOSE", "Figures, traces, report, and saved model"),
            ("OUTPUTS", "report.md + trace CSVs + plots"),
        ]),
        ("Multi-seed run", 840, ORANGE_SOFT, [
            ("ENVIRONMENT", "Continuous"),
            ("FORECAST", "Ground truth"),
            ("ITERATIONS", "100"),
            ("SEEDS", "42, 49, 56, 63, 70"),
            ("PURPOSE", "Final quantitative stability claim"),
            ("OUTPUTS", "summary.csv + experiment_summary.json"),
        ]),
    ]
    for title, x, fill, stats in panels:
        _rr(d, (x, 150, x + 680, 620), fill)
        _text(d, (x + 24, 178), title, FONT_H3, NAVY)
        y = 240
        for idx, (label, value) in enumerate(stats):
            fill_row = WHITE if idx % 2 == 0 else (250, 248, 244)
            _rr(d, (x + 24, y, x + 656, y + 52), fill_row)
            _text(d, (x + 40, y + 15), label, FONT_LABEL, MUTED)
            val_color = TEAL if title.startswith("Multi") and label in {"SEEDS", "OUTPUTS"} else NAVY
            _text(d, (x + 610, y + 15), value, FONT_BODY_SM, val_color, anchor="ra")
            y += 62
    _rr(d, (110, 680, 1490, 785), ORANGE_SOFT, None)
    _text(d, (142, 720), "Takeaway: the project now finishes on one clean task definition: continuous scheduling with reliable forecasts, using one representative artifact run and one five-seed stability run.", FONT_BODY, INK, max_width=1290)
    return img


def slide_14():
    img, d = _new_slide()
    _top_band(d, "What is simplified, and what comes next", "Next steps", "14")
    steps = [
        ("1. Run the artifact seed", "Generate the final figures, traces, report, and saved model from seed 42."),
        ("2. Run five seeds", "Use the same continuous + ground_truth setting for the final stability claim."),
        ("3. Write the result", "Use the artifact run for examples and the multi-seed run for the main numbers."),
        ("4. Keep scope honest", "Frame the result as control quality under reliable forecasts."),
        ("5. Future work", "Reintroduce imperfect forecasts and real-port geography afterward."),
    ]
    xs = [70, 380, 690, 1000, 1310]
    widths = [280, 280, 280, 280, 220]
    fills = [WHITE, TEAL_SOFT, WHITE, ORANGE_SOFT, WHITE]
    for (title, text), x, w, fill in zip(steps, xs, widths, fills):
        _rr(d, (x, 245, x + w, 585), fill)
        _text(d, (x + 18, 270), title, FONT_H3, NAVY)
        _text(d, (x + 18, 335), text, FONT_BODY_SM, INK, max_width=w - 36)
    _text(d, (800, 700), "This finish line is intentionally narrow: complete the project cleanly on the continuous environment, then expand realism afterward.", FONT_BODY, MUTED, anchor="ma")
    return img


def build_pdf() -> Path:
    v3 = _read_v3_metrics()
    slides = [
        slide_1(),
        slide_2(),
        slide_3(),
        slide_4(),
        slide_5(),
        slide_6(),
        slide_7(),
        slide_8(),
        slide_9(),
        slide_10(),
        slide_11(),
        slide_12(v3),
        slide_13(),
        slide_14(),
    ]
    slides[0].save(OUT, "PDF", resolution=150.0, save_all=True, append_images=slides[1:])
    return OUT


if __name__ == "__main__":
    print(build_pdf())
