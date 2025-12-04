# Machine-Learning-Integration-of-FTIR-for-Predicting-BE-in-Pleurotus-pulmonarius-Cultivation
Code and workflow for machine-learning models (Random Forest and Decision Trees) integrating FTIR descriptors to predict Biological Efficiency (BE%) in Pleurotus pulmonarius cultivation.
# Machine Learning Integration of FTIR for Predicting BE in *Pleurotus pulmonarius* Cultivation

Code and workflow for machine-learning models (Random Forest and Decision Trees) integrating FTIR-derived lignocellulosic descriptors to predict Biological Efficiency (BE%) in *Pleurotus pulmonarius* cultivation.  
This pipeline accompanies the manuscript:

**“Machine Learning Integration of FTIR for Predicting BE in *Pleurotus pulmonarius* Cultivation”**  
*(to be submitted)*


## 🔬 Overview of the Pipeline

This repository includes the complete R workflow for:

### **1. FTIR-based predictors**
- C890 — cellulose backbone vibration  
- C1420 — CH₂ deformation / cellulose crystallinity  
- C1510 — lignin aromatic ring  
- C1740 — hemicellulose carbonyl stretch  

### **2. Machine Learning Models**
- **Random Forests (ranger + caret)**  
  - 70/30 train-test split  
  - 10-fold cross-validation  
  - Global CV10 performance  
  - Bootstrap variable importance (B = 500)

- **Decision Trees (rpart)**  
  - Large tree → pruning to ~8 leaves  
  - Nested CV10 with out-of-fold predictions  
  - Inferno-based visual tree representation

### **3. Visualization**
- Heatmaps of predictive surfaces (Top-2 FTIR variables)
- Pruned decision tree diagrams
- Bland–Altman plots including:
  - Bias  
  - 95% LoA  
  - R²  
  - RMSE  
  - MAE  
  - Pearson r and p-value  


## 📁 Repository Structure

