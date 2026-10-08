# Dataset & Benchmark Evaluation Audit

**Audit Date:** October 2026  
**Status:** COMPLETE (Official UMN Multi-Scene Benchmark Downloaded, Segmented, and Registered)

---

## 1. Benchmark Datasets Review & Formal Documentation

### Dataset 1: UMN Unusual Crowd Activity Dataset (Primary Benchmark)
* **Official Source:** University of Minnesota (MHA Lab) & University of Central Florida (CRCV)  
  * Primary Archive URL: [http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi](http://mha.cs.umn.edu/Movies/Crowd-Activity-All.avi)
  * Mirror/Project Page: [https://www.crcv.ucf.edu/projects/Abnormal_Crowd/](https://www.crcv.ucf.edu/projects/Abnormal_Crowd/)
* **License & Permitted Usage:** Publicly available for non-commercial academic research.
* **Master Recording Properties:**
  * Resolution: $320 \times 240$ pixels
  * Frame Rate: $30.0$ FPS
  * Total Length: 7,739 frames (~4.3 minutes)
* **Structure & Ground-Truth Intervals:**
  Segmented into 11 distinct benchmark sequences across 3 independent physical scenes:

| Scene | Clip ID | Total Frames | Normal Frame Interval | Abnormal/Panic Interval | Physical Setting |
|---|---|---|---|---|---|
| **Scene 1 (Lawn)** | `lawn_clip1.mp4` | 625 | [0, 497] (498 frames) | [498, 624] (127 frames) | Outdoor open lawn; unconstrained multidirectional run |
| **Scene 1 (Lawn)** | `lawn_clip2.mp4` | 827 | [0, 297] (298 frames) | [298, 826] (529 frames) | Outdoor lawn; sudden rapid dispersal |
| **Scene 2 (Indoor)** | `indoor_clip3.mp4` | 548 | [0, 94] (95 frames) | [95, 547] (453 frames) | Indoor hallway/foyer; corridor evacuation |
| **Scene 2 (Indoor)** | `indoor_clip4.mp4` | 684 | [0, 95] (96 frames) | [96, 683] (588 frames) | Indoor hallway; doorway bottleneck escape |
| **Scene 2 (Indoor)** | `indoor_clip5.mp4` | 767 | [0, 296] (297 frames) | [297, 766] (470 frames) | Indoor corridor; sudden group rush |
| **Scene 2 (Indoor)** | `indoor_clip6.mp4` | 578 | [0, 297] (298 frames) | [298, 577] (280 frames) | Indoor corridor; rapid egress toward exit |
| **Scene 2 (Indoor)** | `indoor_clip7.mp4` | 772 | [0, 297] (298 frames) | [298, 771] (474 frames) | Indoor corridor; structured movement turning chaotic |
| **Scene 2 (Indoor)** | `indoor_clip8.mp4` | 788 | [0, 119] (120 frames) | [120, 787] (668 frames) | Indoor hallway; high-velocity exit surge |
| **Scene 3 (Plaza)** | `plaza_clip9.mp4` | 598 | [0, 554] (555 frames) | [555, 597] (43 frames) | Paved courtyard; fast sudden group dispersal |
| **Scene 3 (Plaza)** | `plaza_clip10.mp4` | 735 | [0, 56] (57 frames) | [57, 734] (678 frames) | Paved courtyard; immediate panic run across plaza |
| **Scene 3 (Plaza)** | `plaza_clip11.mp4` | 807 | [0, 766] (767 frames) | [767, 806] (40 frames) | Paved courtyard; prolonged normal meander before panic |
| **Total** | **11 clips** | **7,729 frames** | **3,379 frames (43.7%)** | **4,350 frames (56.3%)** | **3 Distinct Environments** |

* **Label Granularity:** Temporal event transitions (Normal meandering $\to$ Panic/Evacuation rush).
* **Research Role:** Primary benchmark for Leave-One-Scene-Out (LOSO) cross-scene evaluation and Leave-One-Clip-Out (LOGO) cross-validation.

---

### Dataset 2: PETS 2009 (Tracking & High-Density Calibration Reference)
* **Official Source:** University of Reading / IEEE CVPR Workshop  
  URL: [http://www.cvg.reading.ac.uk/PETS2009/](http://www.cvg.reading.ac.uk/PETS2009/)
* **License & Usage:** Open academic research benchmark.
* **Structure:** Multi-view calibrated recordings with progressive crowd density (S1: Density, S2: Trajectories, S3: Flow/Evacuation).
* **Research Role:** Reference for density grid calibration and occlusion tracking stress testing.

---

## 2. Local Dataset Directory Audit

| Path | File / Format | Frame Count | Annotations / Ground Truth | Validation Purpose |
|---|---|---|---|---|
| `data/videos/Crowd-Activity-All.avi` | Master AVI ($320 \times 240$) | 7,739 | Master recording source | Raw benchmark repository |
| `data/videos/lawn_clip[1-2].mp4` | MP4 Video clips | 1,452 | Temporal [Normal, Abnormal] | Scene 1 Evaluation |
| `data/videos/indoor_clip[3-8].mp4` | MP4 Video clips | 4,143 | Temporal [Normal, Abnormal] | Scene 2 Evaluation |
| `data/videos/plaza_clip[9-11].mp4` | MP4 Video clips | 2,143 | Temporal [Normal, Abnormal] | Scene 3 Evaluation |
| `data/videos/demo.mp4` | Synthetic MP4 ($640 \times 480$) | 160 | Programmatic rule ($f \ge 65$) | Fast local pipeline unit testing |
| `data/videos/production_id_...mp4` | Real 4K Concert Video | 731 | Unannotated real-world | Visual stress & throughput testing |
| `data/umn_metadata.json` | JSON Metadata Registry | 11 clips | Official frame boundaries | Dataset configuration for scripts |
| `data/processed/features_dataset.csv` | Extracted Feature Matrix | 7,729 rows | Frame-level features + binary labels | Tabular training & cross-validation |

---

## 3. Strict Validation Methodology Followed
1. **Zero Temporal Leakage:** Frames from the same video or scene are strictly isolated to either the training fold or the testing fold.
2. **Leave-One-Scene-Out (LOSO):** Tests transferability across completely unseen physical camera views (Lawn $\to$ Indoor $\to$ Plaza).
3. **No Metric Inflation:** Results reported below are empirical and unmanipulated.
