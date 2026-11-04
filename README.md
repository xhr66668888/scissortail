# Scissortail — Articulatory Kinematics Workbench

Desktop application (Rust + Tauri) for clinical speech research with
electromagnetic articulography (EMA), built around the MAVIS / MVIEW way of
working: **drop the trial files in, the speech is cut out and the landmarks
are marked automatically, and anything that is off is dragged into place.**
The window keeps the look of a classic MATLAB figure — grey figure, bevelled
controls, Helvetica, white axes with black frames — and is black and white
throughout; the only colour is the status dot in front of each file.

For every trial it marks

1. the **speech segment**: onset and offset of the target word (or, without a
   target word, of the main stretch of speech);
2. for every articulator sensor (tongue tip, jaw, …) and dimension, the
   **position peak (P)**, **velocity peak (V)** and **acceleration peak (A)**
   inside that segment.

## Installing

There are four builds of the same program, on the Releases page:

| file | for |
|---|---|
| `Scissortail_0.2.0_macOS-AppleSilicon.dmg` | Macs with Apple silicon (M1 and later), macOS 13.4 or later |
| `Scissortail_0.2.0_macOS-Intel.dmg` | Intel Macs, macOS 10.15 Catalina or later |
| `Scissortail_0.2.0_Windows-x64-setup.exe` | 64-bit Windows 7 SP1, 8.1, 10 and 11 |
| `Scissortail_0.2.0_Windows-ARM64-setup.exe` | Windows 11 on Arm (Snapdragon X and other Arm laptops) |

**macOS**: open the `.dmg` and drag Scissortail into Applications. On another
Mac the first open is refused (“Apple could not verify …”): open **System
Settings → Privacy & Security** and click **Open Anyway** once (macOS 10.15–12:
right-click the app and choose **Open**) — the app is ad-hoc signed, not
notarized. Scissortail used to be called ArticuMark; an installed ArticuMark
can be deleted, as Scissortail takes over its workspace.

**Windows**: run the setup file. It installs Scissortail for the current user
(no administrator rights needed) with a Start-menu entry. The installer is not
code-signed, so SmartScreen may first say “Windows protected your PC”: click
**More info → Run anyway**. Scissortail draws its window with Microsoft Edge
WebView2, which Windows 10 and 11 normally have; otherwise the installer
downloads it — on Windows 7 and 8.1 version 109, the last for those systems.
That needs an internet connection once: on an offline PC, install the WebView2
Runtime 109 first.

## Using it

1. **Drop `.mat` files or whole folders** anywhere in the window (or use the
   two open buttons). Files appear in the list on the left, grouped by folder,
   and are analysed in the background; the file you are looking at is always
   analysed first. The dot in front of each file tells its state:
   **green** = extracted, **yellow** = please check (low confidence),
   **red** = nothing found / file unreadable, **black** = not analysed yet.
   The legend under the list counts them; click an entry to show only those
   files (click again for all).
2. The window shows, on one time axis, the audio, the spectrogram and one panel
   per sensor with its position, velocity and acceleration. The speech
   segment is shaded grey between the **onset and offset lines** (thick black,
   with a flag at the top pointing into the segment). The landmarks are the
   **P / V / A** markers: a solid (P), dashed (V) or dotted (A) line through
   the sensor panel, with its tag above the panel and a dot on its curve —
   white tag and round dot = automatic, black tag and square dot = placed by
   hand, hollow dot = the peak lies on the segment edge.
3. **Drag** the onset or offset line to move the segment — the landmarks are
   recomputed at once. **Drag a P / V / A marker** (or its tag) to move a
   landmark: it snaps to the nearest peak of its curve, or hold **⌥ Option**
   (**Alt** on Windows) to place it freely. **Double-click** a hand-placed marker to put it back.
   **Shift-drag** over the signals to mark a segment by hand.
4. Keys: `Space` play the segment · `←/→` previous / next file · `Enter` mark
   the file checked and go to the next · `Z` (or double-click) zoom to the
   segment / whole file · scroll or pinch to zoom, drag to pan.
5. **Export CSV** writes the landmarks of every analysed file in the list to
   one table.

Everything is saved automatically (hand edits survive restarts and
re-analysis). The table under the signals lists the landmarks of the shown
dimension (`z`, `x`, `y`, `3D`), with times in ms relative to the onset.

**Settings** (⚙): how the segment is found — the *target word* by speech
alignment (default: the “It's a ___ again” /tVCVt/ protocol, tadat / tatat /
tedat / tetat chosen per trial from the file-name codes `ad_`, `at_`, `ed_`,
`et_`), or the *whole speech segment* (no word needed); and the kinematics
filters. Changing the words re-analyses every file that was not edited or
checked by hand.

## Data format

MATLAB Level-5 `.mat` files (the default `save` format; `-v7.3` HDF5 files are
detected and reported). Each file holds an MVIEW-style struct array with fields
`NAME`, `SRATE`, `SIGNAL` (`SOURCE` optional): one `AUDIO` channel and one
element per sensor (`SIGNAL` = samples × [x y z, angles…]). Sensors whose name
contains `REF`, and `UI` (upper incisor, skull-fixed), are treated as reference
sensors and not shown or analysed.
Coordinates (head-corrected AG50x): x = anterior(+), y = lateral, z = superior(+).

## Method

| stage | what happens |
|---|---|
| Target word (default) | MMS wav2vec2 acoustic model (int8 ONNX) → letter posteriors every 20 ms; CTC Viterbi forced alignment of `* carrier-with-target *` (wildcards absorb noise / unrelated speech; variants without leading/trailing carrier words handle recordings that start or stop mid-phrase). Repeated phrases are aligned separately. |
| Speech segment | Without a target word (or when it cannot be aligned): 10-ms energy envelope against an adaptive noise/peak threshold, pauses < 250 ms bridged, blips < 80 ms dropped; the segment with the most energy wins. |
| Boundaries | Onset (and offset, when followed by silence) refined by segmented “hinge” regression on a 1-ms energy envelope. |
| Kinematics | Zero-phase Butterworth low-pass (20 Hz) on position; velocity/acceleration from Savitzky–Golay local cubic regression (40 ms); peaks refined by quadratic regression (sub-sample timing). Peaks on the window edge are flagged. |
| Hand-placed landmarks | Snap to the nearest local extremum within 10 screen pixels (quadratic refinement as above), or stay exactly where dropped (⌥ / Alt). They keep their time when the segment boundaries or the filter settings change; the value is re-read from the signal. |

The alignment model runs with a fixed number of ONNX Runtime threads (2): the
int8 model's posteriors change slightly with the thread count — enough to flip
a few borderline trials — so results must not depend on the machine's cores.

**Engines.** The Apple-silicon build runs the model with ONNX Runtime, the
configuration validated below. ONNX Runtime does not run on Windows 7 and is no
longer published for Intel Macs, so the Intel-Mac and Windows builds run the
same model with tract (pure Rust; one thread per inference, so the number of
cores does not matter there either). On all 1,842 trials of the patient data
(not part of this repository), tract and ONNX Runtime (both on Apple silicon)
found the same onset in 96 % of trials
(within 20 ms: 97 %) and offsets within 20 ms in 89 % (within 50 ms: 96 %), with
the same target word in 99 %. That is somewhat more than ONNX Runtime's own
variation between 2 and 8 threads (onsets 98 %, offsets 94 % within 20 ms):
12.5 % against 7 % of trials moved by more than 20 ms at onset or offset,
mostly at the offset and in borderline trials (29 % of the low-confidence ones,
10 % of the others). Analyse one study on one platform; when results from
different builds are combined, re-check the yellow files.

Validation on the patient data (93 trials, every 20th per session):
onsets within 20 ms of a full-precision MMS reference alignment in 94 % of
trials (98 % within 50 ms; offsets 96 % within 50 ms); the reference spans
contained the two vowel openings expected for /tVCVt/ (two tongue-tip
lowerings) in 83 % of trials. Across all 1,842 trials every trial received a
detection; 11 % were flagged *low confidence* (amber in the file list), mostly
in the post-operative sessions of patients 2 and 5. The plain energy-threshold
epoch used previously (first/last sample above 2 % of the peak) spans the whole
carrier phrase, not the word.

## Export

One CSV, long format — one row per file × segment × sensor × dimension:
segment onset/offset/duration, then for P, V and A the time (s), value, edge
flag and time relative to the onset (ms). Since 0.2 the last columns are
`segment_method` (`phonetic` / `speech`), `pos_peak_manual`,
`vel_peak_manual`, `acc_peak_manual` (placed by hand) and `file`; the earlier
columns keep their order.

## Building

All four installers are built on a Mac with Apple silicon. Requirements: Rust
(stable), the Tauri CLI (`cargo install tauri-cli --version "^2"`) and the
alignment model in `models/` (see below).

```bash
cd Scissortail
scripts/build.sh                 # the four installers → dist/, with the model and SHA256SUMS.txt
scripts/build.sh macos-x64       # or some of: macos-arm64 macos-x64 windows-x64 windows-arm64
```

The app icons are made from the logo, `app/src-tauri/icons/logo.png` (a square
PNG with a transparent background), by `scripts/icons.sh [new logo]`: on a
white rounded tile for macOS, on its own for Windows.

The Windows installers are cross-compiled. The first Windows build downloads
the tools for that into `~/Library/Caches/scissortail-build`: LLVM, cargo-xwin
with Microsoft's C runtime and Windows SDK libraries (building accepts
Microsoft's license for them), NSIS and YY-Thunks. Both Windows builds link the
C runtime statically, so no Visual C++ redistributable is needed. For Windows 7
the x64 build is linked with YY-Thunks, which supplies the few Windows 8/10
functions Rust's standard library calls, and takes the three COM memory
functions that `windows-core` imports from `combase.dll` (Windows 8+) from
`ole32.dll` instead; `scripts/check_win7_imports.py` then checks every import
of the program against Windows 7's export tables (shipped with YY-Thunks). On
Windows itself, `cargo tauri build` in `app/src-tauri` (MSVC toolchain) builds
an installer for Windows 10/11; the Windows 7 support comes from
`scripts/build.sh`.

The macOS bundles are ad-hoc signed (`bundle.macOS.signingIdentity: "-"`), so a
downloaded copy can be opened with *Open Anyway*; without it the app is only
linker-signed and macOS reports a downloaded copy as damaged. Signing with a
Developer ID and notarizing would remove the warning altogether; likewise a
code-signing certificate would remove the SmartScreen warning on Windows.

The speech model runs on ONNX Runtime in the Apple-silicon build and on tract
in the others (see Method); the choice per platform is in
`app/src-tauri/Cargo.toml`. The command-line tool uses ONNX Runtime, or tract
when built with `--no-default-features --features tract`.

The models are too large for the git repository; they go in `models/`. The
app bundles the alignment model only, and every release carries it as a
separate file as well: put `mms-fa-int8.onnx` from the release into `models/`
before building. Its SHA-256 (`shasum -a 256 models/mms-fa-int8.onnx`) is
`bbe26e2aa431b401dcc350611fa13905c4a0acac5d09361bbf6d32cdc492808f`, the file in
the 0.2.0 installers.

| file | size | source / licence |
|---|---|---|
| `mms-fa-int8.onnx` | 355 MB | MMS forced-alignment wav2vec2 (Meta AI, torchaudio `MMS_FA`), exported to ONNX, MatMul int8 — **CC-BY-NC 4.0 (non-commercial use)** |
| `kokoro-v1.0-int8-dur.onnx`, `voices/*.bin` | 92 MB + 0.5 MB each | Kokoro-82M TTS — Apache-2.0; used only by the command-line tool (`synth`, `--engine template`); not published |

Command-line tool (batch / scripting):

```bash
cargo build --release -p scissortail-cli
./target/release/scissortail analyze <data folder> \
    --word "tˈɑdɑt=tadat@ad_" --word "tˈɛdɑt=tedat@ed_" \
    --carrier "ɪts ə * əɡˈɛn" --csv landmarks.csv --out results.json
./target/release/scissortail synth "tˈɑdɑt" --voice af_heart --out tadat.wav
./target/release/scissortail inspect trial.mat
```

Interface development without the desktop shell:
`scissortail-app --dump-fixtures <data folder> <dir>` runs the real pipeline on a
few trials and writes the JSON the interface receives; serve `app/ui` with that
directory and open `index.html?fixtures=<dir>/` in a browser.

## Layout

```
crates/scissortail-core    MAT reader, DSP, IPA, aligner, speech segmentation, TTS, DTW, kinematics, export
crates/scissortail-cli     command-line front end
app/src-tauri              desktop backend (Tauri 2 commands, workspace persistence, background analysis)
app/ui                     single-window front end (HTML/CSS/JS, no build step)
models                     ONNX models and voices (not in git, see Building)
scripts                    build.sh (the four installers), icons.sh (app icons from the logo), Windows 7 import check
```

The workspace (file list, settings, results, edits) is saved automatically in
`~/Library/Application Support/org.scissortail.workbench/workspace.json` on
macOS and `%APPDATA%\org.scissortail.workbench\workspace.json` on Windows.
The program used to be called ArticuMark: on its first start Scissortail copies
ArticuMark's workspace (`org.articumark.workbench`, in the same place), so the
file list and all hand edits carry over.
