# Dense-sheet benchmark fixtures

Three Gupta test sheets (`151`, `216`, `233`) + their labels — the densest sheets where
zero-shot detection collapsed. Committed here as fixtures so `Stage4_DenseSheet_Experiment.ipynb`
can `wget` them into Colab without `drive.mount()` (the VS Code Colab file-upload widget is
non-functional, and Drive mount triggers account-verification friction).

**Note on the `.jpg` files:** the repo `.gitignore` excludes `*.jpg` (no bulk dataset images
in git). These 3 are a deliberate, force-added exception — tiny (1.3MB total) benchmark
fixtures, not the dataset. Gupta PID_Dataset is CC-BY-4.0 (zenodo.org/records/8028570), so
redistributing a few sheets with attribution is permitted.

Do NOT extend this pattern to the full dataset — it stays out of git.
