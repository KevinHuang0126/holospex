# Public surgical video for the ML handoff

The source is **Supplementary Video 2** from Zhou et al. (2024), *Cystic plate
approach in laparoscopic cholecystectomy: a consecutive retrospective analysis*,
doi:10.3389/fsurg.2024.1487568. The article and supplementary material are
distributed under CC BY 4.0. Source footage is separate from the Endoscapes
training data; this clip has no segmentation answer key.

The latest predictions use `small-004-resolution/best.pt` and are in
`ml/outputs/demo-video-small-anatomy/`. They pair with the existing prepared MP4
in `ml/outputs/demo-video-v1/`. The earlier baseline predictions are retained.

- [Publication and license statement](https://pmc.ncbi.nlm.nih.gov/articles/PMC11649668/)
- [CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/)
- [Official downloadable Video 2](https://pmc-oa-opendata.s3.amazonaws.com/PMC11649668.1/Video2.mp4)
- [Official file metadata](https://pmc-oa-opendata.s3.amazonaws.com/PMC11649668.1/PMC11649668.1.json)
- [PMC public cloud access documentation](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)

Original: 31,242,633 bytes, 1280 × 720, about 91 seconds. Downloaded on
2026-09-12. Its MD5 matches the official metadata:
`d93c82ea0b33ab26bc53d0911bb6e620`. SHA-256:
`f5e5fb1a9dc17e7beb4d20856d50dee5005cbe46010b62abafb710d35215e631`.
The full source and provenance JSON are in ignored `ml/data/demo-clip/`.

## Required attribution when showing or sharing this excerpt

© 2024 Zhou, Xiao, Luo, Luo, Tan and Wang. Supplementary Video 2,
doi:10.3389/fsurg.2024.1487568. CC BY 4.0. Source: National Library of Medicine,
PubMed Central Article Datasets on AWS, accessed 2026-09-12. No NIH/NLM or
author endorsement implied. Holospex excerpted the video, removed audio, and
transcoded it; any Holospex overlays are separate, unreviewed ML predictions.

Keep this credit and license link with the clip in the frontend and slides.
Retain the source's existing visual marks; do not call its source footage or
our model outputs a newly reviewed lesson.

## Reproduce the excerpt and predictions

The selection uses source presentation times from 3 through 11 seconds,
chosen for a short continuous view before running the model. Source identity
and clip identity are distinct. FFmpeg is used only for this preparation step;
the inference CLI uses PyAV.

```sh
mkdir -p ml/data/demo-clip ml/outputs/demo-video-v1 ml/outputs/demo-video-small-anatomy
curl --fail --location 'https://pmc-oa-opendata.s3.amazonaws.com/PMC11649668.1/Video2.mp4' --output ml/data/demo-clip/zhou-2024-video2.mp4
shasum -a 256 ml/data/demo-clip/zhou-2024-video2.mp4
ffmpeg -hide_banner -loglevel error -n -i ml/data/demo-clip/zhou-2024-video2.mp4 -map 0:v:0 -an -vf 'trim=start=3:end=11,setpts=PTS-STARTPTS' -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -fps_mode passthrough -map_metadata -1 -movflags +faststart ml/outputs/demo-video-v1/zhou-2024-video2-excerpt.mp4
.venv/bin/python -m holospex_ml predict-video --checkpoint ml/outputs/small-004-resolution/best.pt --input ml/outputs/demo-video-v1/zhou-2024-video2-excerpt.mp4 --media-id zhou-2024-video2-excerpt-3s-11s --device mps --threshold 0.5 --output ml/outputs/demo-video-small-anatomy/predictions.json
.venv/bin/python -m holospex_ml validate ml/outputs/demo-video-small-anatomy/predictions.json
```

Verify the SHA-256 above before using downloaded bytes. Existing inference
outputs are never overwritten; use a fresh run directory for a new export.
The prepared clip contains 120 frames, starts at decoded PTS zero, and retains
the original pixel dimensions. The dataset's assumed 25-fps filename timebase
is irrelevant to this unrelated clip. Use the clip's actual PTS values.

## Integration boundary

The AR/frontend teammate receives the exact MP4, `predictions.json`,
`predictions.info.json`, `media-provenance.json`, and this attribution. Use
`zhou-2024-video2-excerpt-3s-11s` as the media ID. The prediction JSON is an array
of the existing `FrameResult` objects, including an entry when no structures
pass the display filter. Raw argmax masks are in `predictions.masks/`; they
have not undergone the polygon display filter.

The website still uses synthetic fixtures. This handoff does not implement
its video player, reviewed answers, or camera registration. Source-only
selection is not a clinical assessment of procedure stage or safety. A content
reviewer must choose the lesson checkpoints and author CVS feedback.
