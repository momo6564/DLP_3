# Submission checklist

Tick every line before you zip. Most lost marks on this assignment come from this list,
not from low scores. All three questions are compulsory.

## Q1 — object detection

- [ ] Zero-shot YOLOv8 baseline recorded in `runs/q1/results.json`
- [ ] Fine-tuned YOLOv8 recorded
- [ ] Fine-tuned Faster R-CNN recorded
- [ ] Both detectors trained and evaluated on the **same** split (same image counts)
- [ ] Latency for both measured on the same device, and that device is named
- [ ] Per-class AP reported, not only the mean
- [ ] `mAP_small` vs `mAP_large` reported for Faster R-CNN
- [ ] Two qualitative figures: one where YOLO wins, one where Faster R-CNN wins
- [ ] Application analysis: a named domain, its constraints, and your choice justified
      from your own numbers

## Q2 — semantic segmentation

- [ ] At least five runs in `runs/q2/results.json` (baseline, frozen encoder, no
      pretraining, Dice, DeepLabV3)
- [ ] Per-class IoU reported, not only mIoU
- [ ] Class pixel frequencies measured and quoted
- [ ] Qualitative grid included, with two failure cases diagnosed
- [ ] DeepLabV3 compared at the **same** image size and epoch count as the U-Net
- [ ] Application analysis: where boundary accuracy matters and why

## Q3 — RNNs and LSTMs

- [ ] RNN, LSTM and GRU all run at the same hidden size and epochs
- [ ] Sequence-length study at `max_len` 50 / 200 / 500 for at least RNN and LSTM
- [ ] Gradient-flow figure for both RNN and LSTM (`--grad-flow`), on a log axis
- [ ] Both TODO blocks implemented: packing, and the mean-pooling readout
- [ ] Packed vs unpacked numbers reported for at least one cell
- [ ] Parameter counts given for the recurrent layer, not just the whole model
- [ ] Application analysis: two sequence tasks, with the cell you would use for each

## Either way

- [ ] `python starter/make_tables.py` run last, so `report/tables.md` matches your runs
- [ ] Seed stated (42 unless you changed it)

## Report

- [ ] Built from `report/Report_Template.docx`, all italic prompts deleted
- [ ] 10–16 pages, exported as PDF
- [ ] Every figure numbered, captioned, and discussed in the text
- [ ] Every table numbered and referenced
- [ ] Compute statement included (device, GPU time, library versions from `env_check.py`)
- [ ] Each question ends with the "with more compute" paragraph
- [ ] Word limits respected: Q1 ≤ 900, Q2 ≤ 700, Q3 ≤ 700
- [ ] No number in the report that cannot be traced to a file in `runs/`

## Archive

- [ ] Named `<student_id>_DLP_A1.zip`, under 50 MB
- [ ] `report.pdf` at the top level
- [ ] `code/` with your modified starters and `config.yaml`
- [ ] `report/tables.md` and the CSVs
- [ ] `runs/` with `results.json` and figures for all three questions
- [ ] **every `*.pt` checkpoint deleted** (`find runs -name "*.pt" -delete`)
- [ ] `data/` **not** included
- [ ] `AI_use_declaration.md` filled in and included
