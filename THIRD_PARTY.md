# Attribution

The backbone implementations are local reimplementations and augmentation adapters. They are not unmodified upstream packages and do not claim exact reproduction of the upstream training pipelines.

- MIRT and KaNCD: [EduCDM](https://github.com/bigdata-ustc/EduCDM), especially `EduCDM/MIRT/MIRT.py` and `EduCDM/KaNCD/KaNCD.py`.
- KSCD interaction: [InsCD](https://github.com/ECNU-ILOG/InsCD), `inscd/interfunc/kscd.py`.
- NeuralCD: the concept-level monotonic neural response model implemented in `multibackbone/backbones.py`.

KSCD uses independent student, exercise, and concept representations in this adapter. MIRT preserves its multidimensional 2PL response function. Exercise augmentation, ESRG, committee selection, and ADOP are implemented locally.

Check the upstream repositories and applicable rights before redistributing or applying a license to derived implementations. This release does not assign an open-source license on behalf of the authors or upstream rights holders.

