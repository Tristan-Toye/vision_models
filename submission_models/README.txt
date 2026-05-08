Local weights for submission (not committed to git by default)
==============================================================

Populate this folder from your experiment results root, e.g.:

  python zombie_detection/realtime/deployment/sync_submission_models.py \\
      --results-root /media/.../results

Or manually copy each experiment directory listed in
../zombie_detection/realtime/deployment/ACCURACY_TABLE.txt into the matching
subfolder name (rt_detr, yolov11n, ...).

Required layout after sync
---------------------------
  submission_models/rt_detr/          # full rt_detr__... experiment dir
  submission_models/yolov11n/
  submission_models/yolov8n/
  submission_models/heatmap_cnn/      # best_model.pt + train_settings.json
  submission_models/resnet18_head/
  submission_models/template_match/   # template_gray.npy + train_settings.json

Then set active_model in submission_config.yaml (repo root).
