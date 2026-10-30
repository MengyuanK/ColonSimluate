"""
Schematic adult colon for capsule-endoscopy locomotion studies, fitted to
Alazmani et al. (2016) "Quantitative assessment of colorectal morphology:
Implications for robotic colonoscopy", Med Eng Phys 38(2):148-154,
doi:10.1016/j.medengphy.2015.11.018  (24 TCIA CT-colonography patients, supine).

What comes from the paper (supine means):
  * Segment lengths, Table 1 (cm): rectum 23.4, sigmoid 50.6, descending 24.2,
    transverse 57.2, ascending 21.7, cecum 7.8  (total 185.0)
  * Segment diameters, Table 2 (cm): rectum 3.6, sigmoid 2.6, descending 3.3,
    transverse 3.7, ascending 4.5, cecum 4.4  (CO2-insufflated colon)
  * Segment tortuosity, Table 4: mean bend angle (deg) of the centreline,
    measured on a 1 mm-resampled, 40-point moving-average centreline with a
    35 mm span: rectum 42.2, sigmoid 46.9, descending 33.6, transverse 38.7,
    ascending 30.2, cecum 27.1; flexure regions (10 cm): splenic 31.5,
    hepatic 51.8.  Read here as deflection, 0 deg = straight (higher = more
    tortuous, as the paper describes).
  * Landmark definitions (Sec. 2.3): rectosigmoid jn between sacral promontory
    and S3; sigmoid-descending jn at the pelvic brim; hepatic/splenic flexures
    = right/left most cranial inflexion points; ileocaecal valve bounds cecum.

What is NOT from the paper (the paper gives statistics, not 3-D coordinates):
  * The 3-D layout. A textbook "frame" base path is placed by hand, then a
    smooth 3-D wiggle is added inside each segment. Wiggle amplitude and
    wavelength per segment are solved so that each segment's length AND mean
    bend angle match Tables 1 and 4.

Optional second diameter set (uninflated, as a capsule meets the colon):
  abdominal CT, J Clin Med 2023;12:341 (mm): cecum 35.8, ascending 32.6,
  hepatic flexure 26.8, transverse 22.7, splenic flexure 22.2,
  descending 18.2, sigmoid 17.4, rectum 27.1.

Exported frame: DICOM LPS, mm (x = patient left, y = posterior, z = superior),
origin on the midline at iliac-crest level. Outputs go to ./out next to this file.
"""
import os, json, csv, struct
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

SEGMENTS = ["cecum", "ascending", "transverse", "descending", "sigmoid", "rectum"]
ALZ_LEN_CM = dict(cecum=7.8, ascending=21.7, transverse=57.2, descending=24.2, sigmoid=50.6, rectum=23.4)
ALZ_LEN_SD = dict(cecum=2.9, ascending=4.2, transverse=9.3, descending=7.8, sigmoid=13.9, rectum=6.7)
ALZ_DIAM_CM = dict(cecum=4.4, ascending=4.5, transverse=3.7, descending=3.3, sigmoid=2.6, rectum=3.6)
ALZ_ANG = dict(cecum=27.1, ascending=30.2, transverse=38.7, descending=33.6, sigmoid=46.9, rectum=42.2)
ALZ_ANG_SD = dict(cecum=15.5, ascending=17.7, transverse=23.3, descending=21.1, sigmoid=24.2, rectum=27.3)
ALZ_FLEX = dict(hepatic_flexure=(51.8, 22.4), splenic_flexure=(31.5, 18.5))
UNINFL_MM = dict(cecum=35.8, ascending=32.6, hepatic_flexure=26.8, transverse=22.7,
                 splenic_flexure=22.2, descending=18.2, sigmoid=17.4, rectum=27.1)
LM_KEYS = ["cecal_tip", "ileocecal_valve", "hepatic_flexure", "splenic_flexure",
           "descending_sigmoid_jn", "rectosigmoid_jn", "anorectal_jn"]

# ---------------- base "frame" path (design frame: x left, y ANTERIOR, z up) ----------------
def base_points(D, K):
    """Textbook frame. D = depth (mm) of the transverse colon's hang below the
    hepatic flexure; K = height (mm) of the sigmoid loop's apex."""
    P = [((-64, 24, -100), "cecal_tip"), ((-72, 18, -72), None),
         ((-80, 8, -42), "ileocecal_valve"),
         ((-88, -6, 0), None), ((-92, -10, 45), None), ((-90, -8, 88), None), ((-84, 0, 112), None),
         ((-74, 14, 120), "hepatic_flexure")]
    zt = 120.0
    for x, y, f in [(-64, 40, 0.10), (-52, 62, 0.40), (-34, 76, 0.72), (-12, 82, 0.95),
                    (10, 82, 1.00), (32, 76, 0.85), (50, 64, 0.60), (64, 48, 0.32)]:
        P.append(((x, y, zt - f * D), None))
    P += [((76, 28, 130), None), ((86, 12, 156), None),
          ((93, -12, 170), "splenic_flexure"),
          ((99, -32, 152), None), ((102, -38, 112), None), ((102, -36, 70), None),
          ((101, -26, 15), None), ((98, -12, -15), None),
          ((95, 2, -32), "descending_sigmoid_jn")]
    # sigmoid: drops medially into the pelvis, crosses right, then rises to an
    # apex of height K (mm above the pelvic course) and returns posteriorly
    H = K
    for q in [(78, 28, -55), (52, 40, -85), (22, 40, -98), (-8, 32, -88), (-24, 22, -64),
              (-24, 14, -40 + 0.55 * H), (-8, 6, -40 + H), (12, -2, -40 + 0.85 * H),
              (24, -16, -40 + 0.4 * H), (16, -36, -44)]:
        P.append((q, None))
    P += [((4, -50, -44), "rectosigmoid_jn"),
          ((3, -70, -75), None), ((1, -80, -115), None), ((0, -76, -155), None), ((0, -56, -188), None),
          ((0, -30, -205), "anorectal_jn")]
    return P


def catmull_rom(pts, samples=80, alpha=0.5):
    pts = np.asarray(pts, float)
    ext = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    out, kidx = [], [0]
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1:i + 3]
        t1 = np.linalg.norm(p1 - p0) ** alpha
        t2 = t1 + np.linalg.norm(p2 - p1) ** alpha
        t3 = t2 + np.linalg.norm(p3 - p2) ** alpha
        ts = np.linspace(t1, t2, samples, endpoint=False)[:, None]
        A1 = (t1 - ts) / t1 * p0 + ts / t1 * p1
        A2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        A3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        B1 = (t2 - ts) / t2 * A1 + ts / t2 * A2
        B2 = (t3 - ts) / (t3 - t1) * A2 + (ts - t1) / (t3 - t1) * A3
        out.append((t2 - ts) / (t2 - t1) * B1 + (ts - t1) / (t2 - t1) * B2)
        kidx.append(kidx[-1] + samples)
    out.append(pts[-1:])
    return np.vstack(out), kidx


def arclen(c):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])


def resample(c, step=1.0):
    s = arclen(c)
    u = np.append(np.arange(0, s[-1], step), s[-1])
    return np.column_stack([np.interp(u, s, c[:, k]) for k in range(3)]), s, u


def gsmooth(c, sigma):
    h = int(3 * sigma)
    g = np.exp(-0.5 * (np.arange(-h, h + 1) / sigma) ** 2); g /= g.sum()
    pad = np.vstack([2 * c[0] - c[h:0:-1], c, 2 * c[-1] - c[-2:-h - 2:-1]])
    return np.column_stack([np.convolve(pad[:, k], g, mode="valid") for k in range(3)])


def rmf(c):
    """Tangents and rotation-minimising normal/binormal along a polyline."""
    T = np.gradient(c, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.zeros_like(c)
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    N[0] = np.cross(T[0], a); N[0] /= np.linalg.norm(N[0])
    for i in range(1, len(c)):
        v = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
        N[i] = v / np.linalg.norm(v)
    return T, N, np.cross(T, N)


def make_base(D, K):
    P = base_points(D, K)
    dense, kidx = catmull_rom([p for p, _ in P])
    sd = arclen(dense)
    tags = {t: sd[kidx[i]] for i, (_, t) in enumerate(P) if t}
    C, _, U = resample(dense)
    C = gsmooth(C, 6.0)
    T, N, B = rmf(C)
    return dict(C=C, U=U, N=N, B=B, tags=tags)


def build(params):
    """params: [D, K] + per segment (amplitude mm, wavelength mm)."""
    bs = make_base(params[0], params[1])
    BASE_C, BASE_U, BASE_N, BASE_B, BASE_TAGS = bs["C"], bs["U"], bs["N"], bs["B"], bs["tags"]
    params = params[2:]
    off = np.zeros_like(BASE_C)
    for k in range(6):
        A, lam = params[2 * k], params[2 * k + 1]
        s0, s1 = BASE_TAGS[LM_KEYS[k]], BASE_TAGS[LM_KEYS[k + 1]]
        m = (BASE_U >= s0) & (BASE_U <= s1)
        u = BASE_U[m] - s0; L = s1 - s0
        ramp = 25.0
        w = np.clip(np.minimum(u, L - u) / ramp, 0, 1); w = w * w * (3 - 2 * w)
        f1 = np.sin(2 * np.pi * u / lam)
        f2 = 0.65 * np.sin(2 * np.pi * u / (1.6 * lam) + 1.3)
        off[m] = (A * w)[:, None] * (f1[:, None] * BASE_N[m] + f2[:, None] * BASE_B[m])
    c = gsmooth(BASE_C + off, 4.0)
    su = arclen(c)
    tags = {k: float(np.interp(v, BASE_U, su)) for k, v in BASE_TAGS.items()}
    xyz, _, s = resample(c)
    return s, xyz, tags


def bend_angles(xyz, span=35, ma=40):
    """Alazmani method: 1 mm centreline, 40-pt moving average, deflection angle
    between (n-span -> n) and (n -> n+span); 0 deg = straight."""
    k = np.ones(ma) / ma
    c = np.column_stack([np.convolve(np.pad(xyz[:, j], (ma // 2, ma - ma // 2 - 1), mode="edge"), k, mode="valid")
                         for j in range(3)])
    n = len(c); ang = np.full(n, np.nan)
    i = np.arange(span, n - span)
    a = c[i] - c[i - span]; b = c[i + span] - c[i]
    cosv = np.einsum("ij,ij->i", a, b) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    ang[i] = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
    return ang


def metrics(s, xyz, tags):
    ang = bend_angles(xyz)
    L, M = {}, {}
    for k, seg in enumerate(SEGMENTS):
        a, b = tags[LM_KEYS[k]], tags[LM_KEYS[k + 1]]
        L[seg] = (b - a) / 10
        m = (s >= a) & (s <= b) & ~np.isnan(ang)
        M[seg] = float(np.nanmean(ang[m]))
    F = {}
    for key in ("hepatic_flexure", "splenic_flexure"):
        m = (np.abs(s - tags[key]) <= 50) & ~np.isnan(ang)   # 10 cm region, as in Table 4
        F[key] = float(np.nanmean(ang[m]))
    return L, M, F, ang


def residual(p):
    s, xyz, tags = build(p)
    L, M, _, _ = metrics(s, xyz, tags)
    r = []
    for seg in SEGMENTS:
        r.append(2.0 * (L[seg] - ALZ_LEN_CM[seg]) / ALZ_LEN_CM[seg])
        r.append((M[seg] - ALZ_ANG[seg]) / ALZ_ANG[seg])
    _, _, F, _ = metrics(s, xyz, tags)
    for key, (mu, sd) in ALZ_FLEX.items():
        r.append(0.5 * (F[key] - mu) / mu)
    # penalise bends tighter than the (inflated) lumen radius: they fold the wall
    T = np.gradient(xyz, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    kap = np.linalg.norm(np.gradient(T, s, axis=0), axis=1)
    seg_id = np.clip(np.searchsorted([tags[k] for k in LM_KEYS], s, side="right") - 1, 0, 5)
    rad = np.array([ALZ_DIAM_CM[SEGMENTS[i]] * 5 for i in seg_id])
    r.append(3.0 * np.mean(np.clip(kap * rad - 0.8, 0, None)))
    return np.array(r)


if __name__ == "__main__":
    p0 = np.array([130, 60] + [8, 150] * 6, float)
    lo = np.array([60, 0] + [0, 100] * 6, float); hi = np.array([200, 170] + [30, 320] * 6, float)
    fit = least_squares(residual, p0, bounds=(lo, hi), diff_step=1e-3, max_nfev=800)
    P = fit.x
    s, xyz, tags = build(P)
    L, M, F, ang = metrics(s, xyz, tags)
    print("transverse hang D =", round(P[0]), "mm, sigmoid apex height =", round(P[1]), "mm")
    print("wiggle (A mm, lambda mm):", [(round(P[2+2*k], 1), round(P[3+2*k])) for k in range(6)])
    for seg in SEGMENTS:
        print(f"{seg:11s} length {L[seg]:5.1f} cm (Alazmani {ALZ_LEN_CM[seg]})   mean bend {M[seg]:5.1f} deg (Alazmani {ALZ_ANG[seg]})")
    for k, v in F.items():
        print(f"{k:16s} mean bend {v:5.1f} deg (Alazmani {ALZ_FLEX[k][0]} +/- {ALZ_FLEX[k][1]})")
    print("total", round(s[-1] / 10, 1), "cm")

    # ---- to LPS ----
    xyz = xyz * np.array([1.0, -1.0, 1.0])
    bounds = [tags[k] for k in LM_KEYS]
    seg_id = np.clip(np.searchsorted(bounds, s, side="right") - 1, 0, 5)
    mid = lambda a, b: 0.5 * (tags[a] + tags[b])

    def profile(knots):
        ks, kd = zip(*knots)
        d = PchipInterpolator(ks, kd)(s)
        r = d / 2
        R0 = r[0]; cap = s < R0
        r[cap] = np.sqrt(np.maximum(R0 ** 2 - (R0 - s[cap]) ** 2, 0.25))
        return r

    # Alazmani diameters: constant per segment, blended over +/-15 mm at boundaries
    knA = [(tags["cecal_tip"], 44.0)]
    seq = [("cecum", "cecal_tip", "ileocecal_valve"), ("ascending", "ileocecal_valve", "hepatic_flexure"),
           ("transverse", "hepatic_flexure", "splenic_flexure"), ("descending", "splenic_flexure", "descending_sigmoid_jn"),
           ("sigmoid", "descending_sigmoid_jn", "rectosigmoid_jn"), ("rectum", "rectosigmoid_jn", "anorectal_jn")]
    for seg, a, b in seq:
        dmm = ALZ_DIAM_CM[seg] * 10
        lo_, hi_ = tags[a], tags[b]
        knA += [(lo_ + min(15, (hi_ - lo_) / 3), dmm), (hi_ - min(15, (hi_ - lo_) / 3), dmm)]
    knA.append((tags["anorectal_jn"], 36.0 * 0.6))
    rA = profile(sorted(knA))
    knU = [(tags["cecal_tip"], UNINFL_MM["cecum"]), (mid("cecal_tip", "ileocecal_valve"), UNINFL_MM["cecum"]),
           (mid("ileocecal_valve", "hepatic_flexure"), UNINFL_MM["ascending"]), (tags["hepatic_flexure"], UNINFL_MM["hepatic_flexure"]),
           (mid("hepatic_flexure", "splenic_flexure"), UNINFL_MM["transverse"]), (tags["splenic_flexure"], UNINFL_MM["splenic_flexure"]),
           (mid("splenic_flexure", "descending_sigmoid_jn"), UNINFL_MM["descending"]),
           (mid("descending_sigmoid_jn", "rectosigmoid_jn"), UNINFL_MM["sigmoid"]), (tags["rectosigmoid_jn"], UNINFL_MM["sigmoid"]),
           (mid("rectosigmoid_jn", "anorectal_jn"), UNINFL_MM["rectum"]), (tags["anorectal_jn"], UNINFL_MM["rectum"] * 0.6)]
    rU = profile(knU)

    # curvature for the viewer's fit check
    T = np.gradient(xyz, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    kappa = np.linalg.norm(np.gradient(T, s, axis=0), axis=1)
    kappa = np.convolve(kappa, np.ones(7) / 7, mode="same")

    # self-intersection check (uses the larger, inflated radii)
    from scipy.spatial import cKDTree
    worst = (np.inf, 0, 0)
    for i, j in cKDTree(xyz).query_pairs(70.0):
        if abs(s[i] - s[j]) > 150:
            gap = np.linalg.norm(xyz[i] - xyz[j]) - rA[i] - rA[j]
            if gap < worst[0]: worst = (gap, s[i], s[j])
    print("closest approach of non-adjacent walls (inflated):", np.round(worst, 1))
    tight = int(np.sum((kappa * rA > 1.0) & (s > 30)))
    print("points where bend radius < inflated lumen radius:", tight)

    # ---- mesh + STL ----
    def tube(xyz, r, nseg=36):
        _, N, B = rmf(xyz)
        th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        V = (xyz[:, None, :] + r[:, None, None] * (np.cos(th)[None, :, None] * N[:, None, :]
                                                   + np.sin(th)[None, :, None] * B[:, None, :])).reshape(-1, 3)
        n = len(xyz); F = []
        for i in range(n - 1):
            for k in range(nseg):
                a0 = i * nseg + k; a1 = i * nseg + (k + 1) % nseg
                F += [(a0, a0 + nseg, a1), (a1, a0 + nseg, a1 + nseg)]
        c0 = len(V); V = np.vstack([V, xyz[0], xyz[-1]])
        last = (n - 1) * nseg
        for k in range(nseg):
            F += [(c0, k, (k + 1) % nseg), (c0 + 1, last + (k + 1) % nseg, last + k)]
        F = np.array(F)
        tri = V[F]
        vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
        if vol < 0: F = F[:, ::-1]
        return V, F, abs(vol)

    def write_stl(path, V, F, header):
        tri = V[F]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
        rec["n"] = nrm; rec["v"] = tri.reshape(-1, 9)
        with open(path, "wb") as f:
            f.write(header.encode()[:80].ljust(80, b" "))
            f.write(struct.pack("<I", len(F))); f.write(rec.tobytes())

    os.makedirs(OUT, exist_ok=True)
    for tag, r, label in [("inflated_alazmani", rA, "CO2-inflated diam, Alazmani 2016"),
                          ("uninflated", rU, "uninflated diam, J Clin Med 2023")]:
        V, Fc, vol = tube(xyz, r)
        write_stl(os.path.join(OUT, f"colon_{tag}.stl"), V, Fc, f"Schematic colon, {label}; mm, LPS")
        print(f"colon_{tag}.stl  volume {vol/1000:.0f} mL  faces {len(Fc)}")

    with open(os.path.join(OUT, "colon_centreline.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_mm_from_cecal_tip", "x_left_mm", "y_posterior_mm", "z_superior_mm", "segment",
                    "diam_inflated_alazmani_mm", "diam_uninflated_mm", "bend_angle_deg_35mm_span",
                    "radius_of_curvature_mm"])
        for i in range(len(s)):
            w.writerow([round(s[i], 1), *np.round(xyz[i], 2), SEGMENTS[seg_id[i]], round(2 * rA[i], 2),
                        round(2 * rU[i], 2), "" if np.isnan(ang[i]) else round(ang[i], 1),
                        round(min(1 / max(kappa[i], 1e-6), 9999), 1)])

    # landmarks + viewer data
    NOTES = {
        "cecal_tip": ("Cecal tip", "Blind end of the colon and the end of a rectal-entry run."),
        "ileocecal_valve": ("Ileocecal valve", "Opening to the small bowel; boundary between the ascending colon and cecum."),
        "hepatic_flexure": ("Hepatic flexure", "Right-most cranial turn under the liver. Alazmani: most tortuous flexure region, mean bend 51.8° ± 22.4°."),
        "splenic_flexure": ("Splenic flexure", "Left-most cranial turn under the spleen; highest point. Alazmani: mean bend 31.5° ± 18.5°."),
        "descending_sigmoid_jn": ("Descending–sigmoid junction", "At the pelvic brim, where the descending colon angles forward (Alazmani definition)."),
        "rectosigmoid_jn": ("Rectosigmoid junction", "Between the sacral promontory and S3 (Alazmani definition)."),
        "anorectal_jn": ("Anorectal junction", "Entry point for a rectally inserted capsule."),
    }
    lms = []
    for key in LM_KEYS:
        i = min(np.searchsorted(s, tags[key]), len(s) - 1)
        win = (np.abs(s - tags[key]) <= 50) & ~np.isnan(ang)
        e = dict(key=key, name=NOTES[key][0], note=NOTES[key][1], s_mm=round(tags[key], 1),
                 xyz=[round(float(v), 1) for v in xyz[i]])
        if win.any():
            e["max_bend_deg"] = round(float(np.nanmax(ang[win])), 0)
        if key in F:
            e["mean_bend_deg"] = round(F[key], 1); e["alz_mean"] = ALZ_FLEX[key][0]; e["alz_sd"] = ALZ_FLEX[key][1]
        lms.append(e)
    idx = np.arange(0, len(s), 2)
    if idx[-1] != len(s) - 1: idx = np.append(idx, len(s) - 1)
    data = dict(
        s=[round(float(v), 1) for v in s[idx]], p=[[round(float(c), 1) for c in xyz[i]] for i in idx],
        rA=[round(float(rA[i]), 2) for i in idx], rU=[round(float(rU[i]), 2) for i in idx],
        seg=[int(seg_id[i]) for i in idx], k=[round(float(kappa[i]), 5) for i in idx],
        ang=[None if np.isnan(ang[i]) else round(float(ang[i]), 1) for i in idx],
        landmarks=lms, total_cm=round(float(s[-1]) / 10, 1),
        table=[dict(seg=seg, L=round(L[seg], 1), Lalz=ALZ_LEN_CM[seg], Lsd=ALZ_LEN_SD[seg],
                    A=round(M[seg], 1), Aalz=ALZ_ANG[seg], Asd=ALZ_ANG_SD[seg],
                    D=ALZ_DIAM_CM[seg] * 10) for seg in SEGMENTS],
        flex={k: round(v, 1) for k, v in F.items()},
        params=dict(transverse_hang_mm=round(P[0], 1), sigmoid_apex_mm=round(P[1], 1),
                    wiggle=[[round(P[2 + 2 * k], 1), round(P[3 + 2 * k], 1)] for k in range(6)]),
    )
    with open(os.path.join(OUT, "colon_viewer_data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    # rebuild the browser viewer if its template is present
    tpl = os.path.join(HERE, "viewer", "viewer_template.html")
    if os.path.exists(tpl):
        with open(tpl) as f:
            html = f.read().replace("__DATA__", json.dumps(data, separators=(",", ":")))
        with open(os.path.join(HERE, "viewer", "colon_model_viewer.html"), "w") as f:
            f.write(html)
        print("viewer updated: viewer/colon_model_viewer.html")
    print("written to", OUT)
