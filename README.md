# Schematic colon model for capsule locomotion

A 3D model of an average adult colon for designing and testing capsule robots. It provides:

- two STL meshes of the colon
- a centreline sampled every 1 mm, giving diameter and bend angle along the length
- a browser viewer that runs a capsule from the rectum to the cecum and checks whether it fits

## Where the data comes from

**Segment lengths, diameters and tortuosity:** Alazmani A, Hood A, Jayne D, Neville A, Culmer P. *Quantitative assessment of colorectal morphology: implications for robotic colonoscopy.* Med Eng Phys 2016;38(2):148–154. [doi:10.1016/j.medengphy.2015.11.018](https://doi.org/10.1016/j.medengphy.2015.11.018)

The paper measured 24 healthy colons from TCIA CT colonography scans. This model uses its supine means:
- segment lengths (Table 1)
- CO₂-inflated diameters (Table 2)
- mean bend angles over a 35 mm span (Table 4)
- landmark definitions (Sec. 2.3)

**Uninflated diameters (alternative mesh):** non-insufflated abdominal CT, *J Clin Med* 2023;12:341. These are closer to what a capsule meets, since capsule endoscopy does not inflate the colon.

**Not from the literature: the 3D layout.** The paper reports statistics, not coordinates. The path was built in three steps:
1. A textbook "frame" path was placed by hand.
2. Smooth 3D wiggles were added within each segment.
3. A least-squares fit set the wiggle sizes, the hang of the transverse colon and the height of the sigmoid apex, so that each segment's length and mean bend angle match the paper.

Segment statistics therefore match the paper, but the location of any individual bend does not correspond to real anatomy.

| Segment | Length, model / paper (cm) | Mean bend, model / paper |
|---|---|---|
| Cecum | 7.5 / 7.8 ± 2.9 | 28° / 27 ± 16° |
| Ascending | 20.2 / 21.7 ± 4.2 | 32° / 30 ± 18° |
| Transverse | 55.8 / 57.2 ± 9.3 | 36° / 39 ± 23° |
| Descending | 23.6 / 24.2 ± 7.8 | 33° / 34 ± 21° |
| Sigmoid | 49.9 / 50.6 ± 13.9 | 41° / 47 ± 24° |
| Rectum | 23.4 / 23.4 ± 6.7 | 41° / 42 ± 27° |
| **Total** | **180.4 / 185.0 ± 18.3** | |

## Files

| File | Contents |
|---|---|
| `build_colon.py` | Builds and fits the model. Writes everything in `out/` and refreshes the viewer. |
| `out/colon_inflated_alazmani.stl` | Lumen surface with the paper's CO₂-inflated diameters |
| `out/colon_uninflated.stl` | Same centreline with uninflated diameters |
| `out/colon_centreline.csv` | Per mm: position, segment, both diameters, bend angle, bend radius |
| `viewer/colon_model_viewer.html` | Interactive 3D viewer. Open it in a browser; it needs internet to load three.js. |

Units are millimetres, in DICOM LPS axes: x = patient's left, y = posterior, z = superior. The origin is on the midline at iliac-crest level.

## Run

```bash
pip install -r requirements.txt
python build_colon.py
```

## Limitations

- **This is an average colon.** Real colons vary by roughly 15–30% in segment length and ±20° in bend angle.
- **The meshes are idealised.** Each is a round tube with no haustral folds and no wall thickness.
- **The diameters are optimistic for a capsule.** The inflated diameters exceed what a capsule meets, and the uninflated mesh reuses the same length.
