Real-time zombie detection pipelines
=====================================

Submission (recommended)
--------------------------
``submission.py`` reads ``submission_config.yaml`` (repo root) and loads one of
``zombie_detection.realtime.deployment`` classes. Weights live under
``submission_models/<active_model>/`` — populate with::

  python zombie_detection/realtime/deployment/sync_submission_models.py \\
      --results-root /path/to/evaluate_models/output

Benchmark / ad-hoc env-based runs
---------------------------------
For ``python -m zombie_detection.realtime.benchmark`` you can still use env vars
or pass ``--checkpoint``. Set:

  RLK_ZOMBIE_PIPELINE   One of: heatmap_cnn, resnet18_head, yolov8n, yolov11n,
                        rt_detr, hog_svm, template_match
                        (default: heatmap_cnn)

  RLK_ZOMBIE_CHECKPOINT Path to:
                        - PyTorch: experiment directory containing best_model.pt
                          (or the .pt file itself)
                        - YOLO/RT-DETR: run directory .../weights/best.pt or
                          experiment root (best.pt is discovered)
                        - hog_svm: experiment dir with hog_svm.joblib
                        - template_match: experiment dir with template_gray.npy
                          (saved automatically after training)

  RLK_ZOMBIE_DEVICE     auto | cpu | cuda | mps   (default auto)

  RLK_ZOMBIE_CONF       float threshold (default 0.35)

  RLK_ZOMBIE_CONFIG     optional path to custom config.yaml

Suggested 3–5 to benchmark on unknown hardware
-----------------------------------------------
1) template_match   — fastest CPU baseline
2) heatmap_cnn      — small CNN, 360×640
3) resnet18_head    — heavier heatmap head
4) yolov8n          — smaller Ultralytics model
5) yolov11n or rt_detr — pick one for “best detector” stress test

Benchmark CLI
-------------
  python -m zombie_detection.realtime.benchmark \\
      --pipeline heatmap_cnn \\
      --checkpoint /path/to/checkpoint_or_dir \\
      --iterations 200 --warmup 20

``benchmark`` times the detector on **synthetic** frames. For **real** KAZ
pixels with **random archer actions** (no trained policy) and a latency
histogram + mean printed to the console, use::

  cd /path/to/RL-KAZ
  PYTHONPATH=. python -m zombie_detection.realtime.bench_live_env \\
      -n 100 --submission-config submission_config.yaml \\
      --plot timing_live.png

``-n`` is how many RGB frames to time; ``--screen`` enables human render
(slower overall; timing still measures only ``pipeline.detect``).

It prints a small markdown table (``mean_ms``, ``std_ms``) and writes
``<plot_stem>_stats.csv`` next to ``--plot`` (override with ``--table``).

Python API
----------
  from zombie_detection.realtime import create_pipeline, PIPELINE_REGISTRY
  pipe = create_pipeline("yolov8n", checkpoint="/path/to/best.pt")
  boxes = pipe.detect(observation_rgb_uint8)  # (N,4) xywh
