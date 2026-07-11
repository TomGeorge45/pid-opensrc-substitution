# Gupta frozen test set (fixtures)

All 20 Gupta test sheets + labels (the frozen Stage 4 eval split). Committed as fixtures so
the zero-shot / benchmark notebooks can `wget` them into Colab without `drive.mount()` — the
VS Code Colab file-upload widget is non-functional and Drive mount triggers account-
verification friction.

**`.jpg` files are force-added past the repo `.gitignore`** (which excludes `*.jpg`) — a
deliberate exception. 5.5MB total, CC-BY-4.0 (zenodo.org/records/8028570), so redistributing
the test sheets with attribution is permitted. This is the frozen 20-sheet eval set, not the
bulk dataset — the 72 train sheets and everything else stay out of git.
