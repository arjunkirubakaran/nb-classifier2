# Project 2 — Student Naive Bayes Report (Unique Submission)

**Run:** 2025-11-11 02:25:39

## Setup
- Smoothing alpha: **1.0**
- Vocabulary size: **105428**
- Train examples: **9998**
- Test examples: **9999**
- Classes: alt.atheism, comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware, comp.sys.mac.hardware, comp.windows.x, misc.forsale, rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt, sci.electronics, sci.med, sci.space, soc.religion.christian, talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc

## Results
- **Accuracy:** 0.8702
- **Macro F1:** 0.8661
- **Micro F1:** 0.8702

### Confusion Matrix (TSV)


### Per-Class Metrics (JSON)
```json
{
  "alt.atheism": {
    "precision": 0.8117647058823529,
    "recall": 0.828,
    "f1": 0.8198019801980198,
    "support": 500
  },
  "comp.graphics": {
    "precision": 0.7495559502664298,
    "recall": 0.844,
    "f1": 0.7939793038570085,
    "support": 500
  },
  "comp.os.ms-windows.misc": {
    "precision": 0.9368932038834952,
    "recall": 0.386,
    "f1": 0.5467422096317281,
    "support": 500
  },
  "comp.sys.ibm.pc.hardware": {
    "precision": 0.7003257328990228,
    "recall": 0.86,
    "f1": 0.7719928186714543,
    "support": 500
  },
  "comp.sys.mac.hardware": {
    "precision": 0.8206521739130435,
    "recall": 0.906,
    "f1": 0.8612167300380228,
    "support": 500
  },
  "comp.windows.x": {
    "precision": 0.7583892617449665,
    "recall": 0.904,
    "f1": 0.8248175182481752,
    "support": 500
  },
  "misc.forsale": {
    "precision": 0.8907563025210085,
    "recall": 0.848,
    "f1": 0.8688524590163934,
    "support": 500
  },
  "rec.autos": {
    "precision": 0.9183266932270916,
    "recall": 0.922,
    "f1": 0.9201596806387226,
    "support": 500
  },
  "rec.motorcycles": {
    "precision": 0.9656565656565657,
    "recall": 0.956,
    "f1": 0.9608040201005025,
    "support": 500
  },
  "rec.sport.baseball": {
    "precision": 0.9856262833675564,
    "recall": 0.96,
    "f1": 0.972644376899696,
    "support": 500
  },
  "rec.sport.hockey": {
    "precision": 0.960552268244576,
    "recall": 0.974,
    "f1": 0.9672293942403178,
    "support": 500
  },
  "sci.crypt": {
    "precision": 0.9439071566731141,
    "recall": 0.976,
    "f1": 0.9596853490658801,
    "support": 500
  },
  "sci.electronics": {
    "precision": 0.8940936863543788,
    "recall": 0.878,
    "f1": 0.8859737638748738,
    "support": 500
  },
  "sci.med": {
    "precision": 0.9747368421052631,
    "recall": 0.926,
    "f1": 0.9497435897435899,
    "support": 500
  },
  "sci.space": {
    "precision": 0.9676113360323887,
    "recall": 0.956,
    "f1": 0.9617706237424547,
    "support": 500
  },
  "soc.religion.christian": {
    "precision": 0.9555984555984556,
    "recall": 0.9919839679358717,
    "f1": 0.9734513274336284,
    "support": 499
  },
  "talk.politics.guns": {
    "precision": 0.8956692913385826,
    "recall": 0.91,
    "f1": 0.9027777777777779,
    "support": 500
  },
  "talk.politics.mideast": {
    "precision": 0.9388560157790927,
    "recall": 0.952,
    "f1": 0.945382323733863,
    "support": 500
  },
  "talk.politics.misc": {
    "precision": 0.7495327102803738,
    "recall": 0.802,
    "f1": 0.77487922705314,
    "support": 500
  },
  "talk.religion.misc": {
    "precision": 0.6995515695067265,
    "recall": 0.624,
    "f1": 0.6596194503171248,
    "support": 500
  }
}


