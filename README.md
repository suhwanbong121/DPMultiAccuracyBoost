# Differentially Private Multiaccuracy Boosting

This repository was developed as a CS2260 course project by **Suhwan Bong**, **Sarra Guezguez**, and **Annabel Lowe**.

The project implements **differentially private multiaccuracy post-processing** for classification, as a modification and extension of:

> Kim et al., *Multiaccuracy: Black-Box Post-Processing for Fairness in Classification* (AIES 2019).

Our code builds on the original MultiAccuracyBoost implementation and augments it with:
- a **DP-SGD–based linear auditor**, and  
- a **differentially private multiaccuracy boosting loop** with per-round privacy accounting,

in order to study the trade-offs between **fairness**, **accuracy**, and **differential privacy** on the LFW+A benchmark.
