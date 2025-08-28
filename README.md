
📅 Progress Log (First 3 Weeks)

Week 1 – Dataset Acquisition & Initial Understanding

Acquired two datasets:

1. Mario Dataset — Gameplay logs from 74 human players across 11 levels, containing detailed event timelines.


2. Runebound Depths Synthetic Dataset — Internally generated labelled trajectories simulating various playstyles (e.g., runner, aggressor, explorer, magician).



Studied dataset structures and folder organization to understand event encoding and session structure.


---

Week 2 – Trajectory Construction & Preprocessing

Designed a trajectory representation for sequential modelling:

Features: jumps, kills, coins, powerups, deaths, and movement counts per timestep.

Each trajectory corresponds to a single playthrough sequence.


Wrote a preprocessing pipeline to:

Read all player-level CSV logs from the Mario dataset

Convert raw event logs into cumulative event-count sequences

Combine all trajectories into a single file (combined_trajectories.csv)


Applied similar formatting to the Runebound Depths dataset to ensure both datasets have a consistent input structure.



---

Week 3 – Data Formatting & First Model Implementation

Converted both datasets into .npy format for faster loading:

mario_sequences_augmented.npy (~7,000 sequences)

runebound_depths_synthetic_10000.npy (10,000 labelled sequences)


Implemented the LSTM Autoencoder in PyTorch:

Encodes sequences into a low-dimensional latent space

Reconstructs input sequences for unsupervised feature learning


Applied K-Means and Gaussian Mixture Models (GMM) to latent vectors for clustering.

Evaluated clustering performance using:

Silhouette Score

Davies–Bouldin Index

Latent space visualization via PCA and t-SNE


Established an evaluation pipeline that will be reused for all four models for fair comparison.



---
## Sample of Datasets

**CSV sample of raw datasets:**
**CSV sample of raw  mario dataset:**
![CSV Sample](MarioPCGStudy/AnonymizedDirectory/0037499/csv_sample.png)
**CSV sample of raw Runebound dataset:**
![CSV Sample](runebound/csv_sample.png)

**CSV sample of processed Mario dataset:**
![CSV Sample](MarioPCGStudy/csv_sample_489_499.png)


![CSV Sample](latent_space_visualizations.png)

---

✅ Next Steps:

Implement VAE, LSTM-VAE, and Transformer-based Autoencoder.

Train and evaluate all models on both Mario and Runebound Depths datasets.


Compare clustering quality, reconstruction accuracy, and latent space structure.

# Week 4 -5 Progress Report

## 📌 Pivot from Synthetic Dataset to Reinforcement Learning
This week I made a major pivot in the project direction.  
Instead of continuing with the **supervised synthetic dataset**, I took advice and shifted to a **reinforcement learning (RL)** approach.  
RL aligns much better with the problem: training agents to learn behaviors dynamically in an environment rather than fitting on pre-labeled data.

---

## 🏗️ Environment Setup
- Implemented a **custom Gymnasium environment** (`RuneboundDepthsEnv`) to support multiple playstyles:
  - **Runner**
  - **Explorer**
  - **Magician**
  - **Aggressor**
- Added **Pygame visualization** for debugging and monitoring agent behavior.  
  This has been very helpful for checking whether the policies being learned actually match my intentions.

---

## 🤖 Agent Training Results
Using **PPO from Stable-Baselines3**, I trained agents for each playstyle:

- ✅ **Runner**  
  Performs strongly. Learns to reliably reach the exit — reward shaping here is working well.

- ✅ **Explorer**  
  Does reasonably well. Explores sufficiently and provides a good foundation for the **next step: trajectory creation**.

- ⚠️ **Magician**  
  Struggles to learn meaningful behavior. Current shaping isn’t enough to guide it toward powerups.

- ⚠️ **Aggressor**  
  Also underperforming. Fails to consistently seek out or eliminate enemies despite reward incentives.

---

## 🚀 Next Steps
- Continue development with **Runner** and **Explorer** agents (enough progress to move into trajectory generation).  
- **Defer Magician & Aggressor** for now to avoid stalling progress.  
- Revisit these two playstyles later once the pipeline is more mature and other pieces are in place.

---

## 🔎 Summary
Week 4-5 was a key turning point:  
- Pivoted from probability and randomly generated synthetic data → reinforcement learning.  
- Built a working Gym environment with Pygame-based visualization.  
- Trained agents with mixed results:
  - Runner and Explorer: ✅ usable progress
  - Magician and Aggressor: ⚠️ not yet working  
- Clear path forward: focus on what works now.
