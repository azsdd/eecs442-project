# Data included in this repository

- `raw/realwaste/extracted/realwaste-main/RealWaste/<Material>/*.jpg`: the 905 RealWaste images that appear in
  `reviewed/realwaste_candidates.csv`, `reviewed/realwaste_extra_candidates.csv` and `reviewed/splits.csv`
  (660 benchmark images + extra-annotation candidates + reviewed-but-excluded candidates). They are a subset of
  RealWaste (Single, Iranmanesh, Raad, 2023; UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/908/realwaste),
  licensed CC BY-NC-SA 4.0, redistributed here for non-commercial academic use with attribution. The full dataset
  (4,752 images, 656 MB zip) is not included; download it from the link above to rerun the sampling scripts.
- `raw/ood/images/<class>/*.jpg`: the 194 proxy "new photo" candidates (142 kept) extracted from the Hugging Face
  dataset `NeoAivara/Waste_Classification_data` (Apache-2.0), resized to at most 640 px. The 257 MB parquet is not
  included.
- `reviewed/`: our own annotations (candidate lists, review decisions, notes, splits), `review_sheets/`: the contact
  sheets used for the visual review, `masks/`: GrabCut foreground masks used by the background-swap augmentation.
