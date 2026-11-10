---
library_name: transformers
tags:
  - eeg
  - neuroscience
  - foundation-model
  - pytorch
license: other
license_name: reve-responsible-use-license-v1.0
license_link: LICENSE

model-index:
- name: Reve-base
  results:
  - task:
      type: feature-extraction
    dataset:
      name: TUAB
      type: TUAB
    metrics:
    - type: Accuracy
      value: 0.8315
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: TUEV
      type: TUEV
    metrics:
    - type: Accuracy
      value: 0.6759
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: PhysionetMI
      type: PhysionetMI
    metrics:
    - type: Accuracy
      value: 0.648
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCICIV2a
      type: BCICIV2a
    metrics:
    - type: Accuracy
      value: 0.6396
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: FACED
      type: FACED
    metrics:
    - type: Accuracy
      value: 0.5646
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: ISRUC
      type: ISRUC
    metrics:
    - type: Accuracy
      value: 0.7819
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: Mumtaz
      type: Mumtaz
    metrics:
    - type: Accuracy
      value: 0.9644
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: MentalArithmetic
      type: MentalArithmetic
    metrics:
    - type: Accuracy
      value: 0.766
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCI2020-3
      type: BCI2020-3
    metrics:
    - type: Accuracy
      value: 0.5635
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: TUAB-LP
      type: TUAB-LP
    metrics:
    - type: Accuracy
      value: 0.81
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: TUEV-LP
      type: TUEV-LP
    metrics:
    - type: Accuracy
      value: 0.592
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: PhysionetMI-LP
      type: PhysionetMI-LP
    metrics:
    - type: Accuracy
      value: 0.537
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCICIV2a-LP
      type: BCICIV2a-LP
    metrics:
    - type: Accuracy
      value: 0.517
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: ISRUC-LP
      type: ISRUC-LP
    metrics:
    - type: Accuracy
      value: 0.697
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: Mumtaz-LP
      type: Mumtaz-LP
    metrics:
    - type: Accuracy
      value: 0.962
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: MentalArithmetic-LP
      type: MentalArithmetic-LP
    metrics:
    - type: Accuracy
      value: 0.74
      name: Balanced accuracy
  - task:
      type: feature-extraction
    dataset:
      name: BCII2020-3-LP
      type: BCII2020-3-LP
    metrics:
    - type: Accuracy
      value: 0.39
      name: Balanced accuracy
---
# Model Card for REVE-base

<!-- Provide a quick summary of what the model is/does. -->

REVE ([project page here](https://brain-bzh.github.io/reve/)) is a transformer-based foundation model for EEG signal processing. It was trained on 60k hours of EEG data from various sources and is designed to be adaptable to any electrode configuration and a wide range of EEG-based tasks.




## Model Details

### Architecture

<!-- Provide a longer summary of what this model is. -->

REVE (Representation for EEG with Versatile Embeddings), a pretrained encoder explicitly designed to generalize across diverse EEG signals. 
REVE introduces a novel 4D positional encoding scheme that enables it to process signals of arbitrary length and electrode arrangement.
Using a masked autoencoding objective, we pretrain REVE on over 60,000 hours of EEG data from 92 datasets spanning 25,000 subjects.

**Developed by** the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) and [UdeM](https://www.umontreal.ca/en/)

**Funded by:** This research was supported by the French National Research Agency (ANR) through its AI@IMT program and grant ANR-24-CE23-7365, as well as by a grant from the Brittany region. Further support was provided by a Discovery Grant from the Natural Sciences and Engineering Research Council of Canada (NSERC), by funding from the Canada Research Chairs program and the Fonds de recherche du Québec – Nature et technologies (FRQ-NT). This work was granted access to the HPC resources of IDRIS under the allocation 2024-AD011015237R1 made by GENCI, as well as HPC provided by Digital Alliance Canada. 

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [github](https://brain-bzh.github.io/reve/)
- **Paper :** [arxiv](https://arxiv.org/abs/2510.21585)

## Uses

Example script to extract embeddings with REVE, using our position bank:
```python
from transformers import AutoModel

pos_bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
model = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True)

eeg_data = ... # EEG data as a torch Tensor (batch_size, channels, time_points), must be sampled at 200 Hz

electrode_names = [...] # List of electrode names corresponding to the channels in eeg_data
positions = pos_bank(electrode_names) # Get positions (channels, 3)
# Expand the positions vector to match the batch size 
positions = positions.expand(eeg_data.size(0), -1, -1)  # (batch_size, channels, 3)

output = model(eeg_data, positions)
```

## License and Responsible Use

REVE is available for research, commercial, educational, and personal use under the **REVE Responsible Use License v1.0**.

By downloading, using, modifying, redistributing, or deploying REVE or a derivative model, you agree to comply with the terms of the [`LICENSE`](./LICENSE).

In particular, the License prohibits privacy intrusion and re-identification, non-consensual surveillance or profiling, discriminatory or harmful uses, and other uses that violate applicable law. Redistribution and publication of fine-tuned or otherwise modified REVE models are permitted, provided that the License, attribution, and provenance requirements are preserved.