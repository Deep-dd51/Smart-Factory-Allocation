\# 🏭 Smart Factory Allocation



\### Factory Reallocation \& Shipping Optimization Recommendation System



A data-driven decision intelligence system that analyzes historical shipping performance, evaluates alternative factory allocations, predicts operational outcomes, and recommends product-to-factory reallocations to improve shipping efficiency.



🔗 \*\*Live Dashboard:\*\*  

https://smart-factory-allocation-ayfzbxjs8ju9grfdgddft.streamlit.app/



🔗 \*\*GitHub Repository:\*\*  

https://github.com/Deep-dd51/Smart-Factory-Allocation



\---



\## 📌 Project Overview



Manufacturing and distribution organizations often assign products to factories using static or historical allocation rules.



These allocations may result in:



\- Longer shipping lead times

\- Inefficient factory-product assignments

\- Increased operational costs

\- Reduced profitability

\- Underutilization of suitable factory locations

\- Difficulty identifying better allocation alternatives



The \*\*Smart Factory Allocation\*\* system addresses this problem by combining machine learning, historical performance analysis, optimization, and decision intelligence.



The system evaluates alternative factory locations for products and generates ranked recommendations based on operational improvement, historical evidence, reliability, risk, and decision scores.



\---



\## 🎯 Project Objectives



The primary objectives of the project are:



1\. Analyze historical factory and shipping performance.

2\. Identify inefficient product-factory assignments.

3\. Reconstruct reliable shipping lead-time information.

4\. Engineer operational and business-related features.

5\. Build machine-learning models for lead-time prediction.

6\. Evaluate alternative factory locations.

7\. Quantify the potential operational impact of reallocations.

8\. Rank factory reallocation opportunities.

9\. Generate actionable recommendations.

10\. Provide an interactive executive dashboard.



\---



\# 🚀 Key Features



\## 1. Data Audit \& Preprocessing



The system performs:



\- Dataset validation

\- Missing-value analysis

\- Duplicate detection

\- Date consistency analysis

\- Shipping offset investigation

\- Lead-time reconstruction

\- Data cleaning



The original dataset contains approximately \*\*10,194 records\*\*.



\---



\## 2. Feature Engineering



The project creates operational and business features including:



\- Order year

\- Order month

\- Order quarter

\- Order day

\- Day of week

\- Weekend indicator

\- Profit margin

\- Sales per unit

\- Cost per unit

\- Profit per unit

\- Cost ratio

\- Lead-time category

\- Product family

\- Factory/location identifiers



The engineered dataset contains \*\*35 columns\*\*.



\---



\# 🤖 Machine Learning



The project evaluates multiple regression models for shipping lead-time prediction.



\### Models evaluated



\- Linear Regression

\- Random Forest Regressor

\- Gradient Boosting Regressor



\### Model Performance



| Model | MAE | RMSE | R² |

|---|---:|---:|---:|

| Linear Regression | 0.8343 | 1.0510 | 0.6575 |

| Random Forest | \*\*0.6277\*\* | \*\*0.8272\*\* | \*\*0.7878\*\* |

| Gradient Boosting | 0.8512 | 1.0356 | 0.6675 |



\### Selected Model



\*\*Random Forest Regressor\*\*



The Random Forest model achieved the strongest overall performance:



\- \*\*MAE:\*\* 0.6277 days

\- \*\*RMSE:\*\* 0.8272 days

\- \*\*R²:\*\* 0.7878



Therefore, Random Forest was selected as the final lead-time prediction model.



\---



\# 🏭 Factory Reallocation Optimization



After the predictive modeling stage, the system evaluates alternative factory locations for individual products.



For each candidate allocation, the system considers factors such as:



\- Current average lead time

\- Candidate average lead time

\- Lead-time improvement

\- Improvement percentage

\- Historical records

\- Sales

\- Profit

\- Profit difference

\- Reliability

\- Risk

\- Operational benefit

\- Decision score

\- Optimization score



The recommendations are then ranked according to their overall operational value.



\---



\# 📊 Recommendation System



The final recommendation dataset contains information including:



\- `product\_id`

\- `candidate\_location`

\- `current\_avg\_lead\_time`

\- `candidate\_avg\_lead\_time`

\- `lead\_time\_improvement\_days`

\- `lead\_time\_improvement\_pct`

\- `reliability\_score`

\- `candidate\_score`

\- `decision\_score`

\- `risk\_level`

\- `recommendation\_type`

\- `priority`

\- `confidence`

\- `recommended\_action`

\- `decision\_explanation`



\---



\# 📈 Example Recommendations



The current recommendation analysis identified several opportunities.



\### Top recommendation



\*\*Product:\*\* `CHO-SCR-58000`



\*\*Recommended location:\*\* `Massachusetts\_Atlantic`



Historical analysis indicates:



\- Current average lead time: approximately \*\*4.25 days\*\*

\- Candidate average lead time: \*\*3.13 days\*\*

\- Estimated reduction: approximately \*\*1.12 days\*\*

\- Estimated improvement: \*\*26.45%\*\*

\- Historical supporting records: \*\*24\*\*

\- Priority: \*\*STRONG PRIORITY\*\*

\- Confidence: \*\*MEDIUM\*\*

\- Recommended action: \*\*Pilot Reallocation\*\*



\---



\### Other high-potential recommendations



| Product | Recommended Location | Improvement |

|---|---|---:|

| CHO-SCR-58000 | Massachusetts\_Atlantic | 26.45% |

| CHO-MIL-31000 | Ohio\_Atlantic | 13.52% |

| CHO-TRI-54000 | Ohio\_Atlantic | 11.67% |

| CHO-FUD-51000 | Indiana\_Interior | 12.04% |

| OTH-KAZ-38000 | California\_Pacific | 6.01% |

| CHO-NUT-13000 | California\_Pacific | 4.20% |



These recommendations should be interpreted as decision-support outputs based on historical evidence rather than guaranteed future outcomes.



\---



\# 📊 Dashboard



The project includes an interactive \*\*Streamlit dashboard\*\*.



\## Executive Dashboard



The dashboard currently provides an executive-level view of the optimization results.



Current results include:



\- \*\*6 recommendations\*\*

\- \*\*6 products\*\*

\- \*\*4 candidate locations\*\*

\- \*\*12.31% average improvement\*\*

\- \*\*0.53 average days reduced\*\*

\- \*\*213.98 estimated total days saved\*\*

\- \*\*26.45% maximum improvement\*\*

\- \*\*1.12 days maximum average lead-time reduction\*\*



\---



\## Dashboard Sections



\### 1. Executive Dashboard



Provides:



\- Overall recommendation count

\- Product count

\- Candidate location count

\- Average improvement

\- Average days reduced

\- Estimated total days saved

\- Maximum improvement

\- Best recommendation



\### 2. Recommendations



Displays the ranked factory reallocation recommendations.



\### 3. Lead Time Analysis



Compares:



\- Current lead time

\- Recommended lead time

\- Improvement percentage



\### 4. Impact Analysis



Analyzes:



\- Decision scores

\- Recommendation priorities

\- Location impact

\- Product impact

\- Estimated days saved

\- Operational impact



\### 5. Data Explorer



Provides access to the complete recommendation dataset and allows users to download the results as CSV.



\---



\# 🏗️ System Architecture



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Raw Data          │

&#x20;                   │ Nassau Candy Data   │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Data Audit \&        │

&#x20;                   │ Preprocessing       │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Lead-Time           │

&#x20;                   │ Reconstruction     │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Feature Engineering │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ ML Model Training   │

&#x20;                   │                     │

&#x20;                   │ Linear Regression   │

&#x20;                   │ Random Forest       │

&#x20;                   │ Gradient Boosting   │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Lead-Time Prediction │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Factory Location    │

&#x20;                   │ Discovery           │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Allocation Candidate │

&#x20;                   │ Analysis             │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Optimization \&      │

&#x20;                   │ Decision Scoring    │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Final Recommendations│

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │ Streamlit Dashboard │

&#x20;                   └─────────────────────┘

```



\---



\# 📁 Project Structure



```text

Smart-Factory-Allocation/

│

├── app/

│   └── app.py

│

├── config/

│   ├── \_\_init\_\_.py

│   └── settings.py

│

├── data/

│   └── processed/

│       ├── cleaned\_nassau\_candy.csv

│       ├── featured\_nassau\_candy.csv

│       │

│       └── factory\_allocation/

│           ├── allocation\_candidates.csv

│           ├── allocation\_impact\_analysis.csv

│           ├── location\_impact\_summary.csv

│           ├── product\_impact\_summary.csv

│           ├── product\_location\_performance.csv

│           │

│           ├── allocation\_optimization/

│           │   ├── allocation\_optimization.csv

│           │   ├── final\_allocation\_recommendations.csv

│           │   ├── optimization\_location\_summary.csv

│           │   └── optimization\_product\_summary.csv

│           │

│           ├── recommendations/

│           │   ├── executive\_summary.txt

│           │   ├── final\_recommendations.csv

│           │   ├── location\_recommendations.csv

│           │   └── product\_recommendations.csv

│           │

│           └── visualizations/

│               ├── 01\_lead\_time\_comparison.png

│               ├── 02\_improvement\_percentage.png

│               ├── 03\_decision\_score.png

│               ├── 04\_recommendation\_priority.png

│               ├── 05\_location\_impact.png

│               ├── 06\_product\_impact.png

│               ├── 07\_days\_saved.png

│               ├── 08\_executive\_dashboard.png

│               ├── location\_visualization\_summary.csv

│               ├── product\_visualization\_summary.csv

│               └── visualization\_dataset.csv

│

├── notebooks/

│   ├── 01\_Data\_Audit.py

│   ├── 02\_Date\_Investigation.py

│   ├── 03\_Shipping\_Offset\_Analysis.py

│   ├── 04\_Lead\_Time\_Reconstruction.py

│   ├── 05\_Test\_Preprocessing.py

│   ├── 06\_Feature\_Inspection.py

│   ├── 07\_Feature\_Engineering\_Test.py

│   ├── 08\_Model\_Feature\_Audit.py

│   ├── 09\_Save\_Featured\_Data.py

│   ├── 10\_Model\_Preparation.py

│   ├── 11\_Baseline\_Model.py

│   ├── 12\_Random\_Forest\_Model.py

│   ├── 13\_Gradient\_Boosting\_Model.py

│   ├── 14\_Model\_Comparison.py

│   ├── 15\_Train\_Final\_Model.py

│   ├── 16\_Test\_Prediction\_Service.py

│   ├── 17\_Factory\_Allocation\_Audit.py

│   ├── 18\_Factory\_Location\_Discovery.py

│   ├── 19\_Factory\_Allocation\_Candidate\_Analysis.py

│   ├── 20\_Allocation\_Impact\_Analysis.py

│   ├── 21\_Allocation\_Optimization.py

│   ├── 22\_Recommendation\_Decision\_Report.py

│   └── 23\_Allocation\_Visualization.py

│

├── src/

│   ├── data/

│   │   ├── data\_loader.py

│   │   ├── feature\_engineering.py

│   │   ├── preprocessing.py

│   │   └── process\_data.py

│   │

│   ├── models/

│   │   ├── evaluation.py

│   │   ├── model\_training.py

│   │   └── prediction.py

│   │

│   ├── optimization/

│   │   ├── optimizer.py

│   │   └── recommender.py

│   │

│   ├── prediction/

│   │   └── predictor.py

│   │

│   ├── utils/

│   │   └── helpers.py

│   │

│   └── visualization/

│       └── visualization.py

│

├── tests/

│   ├── test\_models.py

│   ├── test\_optimizer.py

│   └── test\_preprocessing.py

│

├── main.py

├── requirements.txt

├── README.md

├── LICENSE

└── .gitignore

```



\---



\# 🛠️ Technologies Used



\### Programming



\- Python



\### Data Processing



\- Pandas

\- NumPy

\- SciPy



\### Machine Learning



\- Scikit-learn

\- Random Forest

\- Gradient Boosting

\- Linear Regression



\### Visualization



\- Matplotlib

\- Plotly

\- Streamlit



\### Dashboard



\- Streamlit



\### Development



\- Git

\- GitHub

\- Python Virtual Environment



\---



\# ⚙️ Installation



\## 1. Clone the repository



```bash

git clone https://github.com/Deep-dd51/Smart-Factory-Allocation.git

cd Smart-Factory-Allocation

```



\## 2. Create a virtual environment



\### Windows



```powershell

python -m venv .venv

```



Activate it:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### Linux / macOS



```bash

python3 -m venv .venv

source .venv/bin/activate

```



\---



\## 3. Install dependencies



```bash

pip install -r requirements.txt

```



The dashboard currently uses:



```text

streamlit==1.61.0

pandas==3.0.5

```



\---



\# ▶️ Running the Dashboard Locally



From the project root:



```bash

streamlit run app/app.py

```



The dashboard will be available at:



```text

http://localhost:8501

```



\---



\# 🌐 Live Deployment



The project is deployed using Streamlit Community Cloud.



\### Live application



https://smart-factory-allocation-ayfzbxjs8ju9grfdgddft.streamlit.app/



\### Deployment configuration



```text

Repository:

Deep-dd51/Smart-Factory-Allocation



Branch:

main



Main file:

app/app.py

```



\---



\# 🧪 Testing



The project includes tests for:



\- Data preprocessing

\- Model functionality

\- Factory allocation optimization



Run the test suite with:



```bash

pytest

```



\---



\# 📌 Business Impact



The system converts historical operational data into actionable factory allocation recommendations.



Instead of simply answering:



> "What happened?"



the system attempts to answer:



> \*\*"What should we change to improve performance?"\*\*



The recommendation pipeline provides decision-makers with:



\- Candidate factory locations

\- Expected lead-time improvement

\- Improvement percentage

\- Historical evidence

\- Risk level

\- Confidence level

\- Operational priority

\- Recommended action

\- Estimated days saved



This enables factory allocation decisions to be evaluated quantitatively rather than relying exclusively on static allocation rules.



\---



\# ⚠️ Important Considerations



The recommendations are \*\*decision-support outputs based on historical data\*\*.



Estimated improvements should not be interpreted as guaranteed future performance.



Before implementing a large-scale factory reassignment, recommendations should ideally be validated through:



1\. Operational review

2\. Capacity analysis

3\. Cost analysis

4\. Pilot reallocation

5\. Supply-chain constraints

6\. Business stakeholder approval



This is particularly important for recommendations with limited historical records or lower confidence.



\---



\# 🔮 Future Improvements



Potential future extensions include:



\- Real-time factory capacity integration

\- Dynamic shipping-cost optimization

\- Inventory constraints

\- Factory capacity constraints

\- Transportation cost modeling

\- Multi-objective optimization

\- Profit-aware allocation optimization

\- Scenario simulation

\- What-if analysis

\- Interactive factory maps

\- Automated recommendation refresh

\- Real-time order integration

\- Explainable AI for allocation decisions

\- Cloud database integration

\- Role-based dashboard access



\---



\# 👨‍💻 Author



\*\*Deep Das\*\*



Computer Science Engineering  

Artificial Intelligence \& Machine Learning



\---



\# 📜 License



This project is licensed under the terms specified in the repository's `LICENSE` file.



\---



\## ⭐ Project Summary



\*\*Smart Factory Allocation\*\* combines:



```text

Data Engineering

&#x20;      +

Machine Learning

&#x20;      +

Optimization

&#x20;      +

Decision Intelligence

&#x20;      +

Interactive Visualization

```



to create a practical recommendation system for factory-product allocation and shipping optimization.

