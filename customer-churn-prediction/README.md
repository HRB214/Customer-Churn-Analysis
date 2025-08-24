# **Predicting and Preventing Customer Churn: An End-to-End Machine Learning Project for Your Portfolio**

## **I. Project Overview and Business Context**

### **1.1 The Business Imperative: Understanding Customer Churn**

In the contemporary business landscape, particularly within subscription-based industries, customer retention has emerged as a paramount indicator of long-term viability and success. Customer churn, also known as customer attrition, is the metric that quantifies the rate at which customers discontinue their relationship with a company or service over a specified period. This metric is not merely a statistical figure; it is a critical barometer of business health, reflecting customer satisfaction, product-market fit, and competitive standing. High churn rates can precipitate a cascade of negative business outcomes, including diminished revenue streams, a decline in market share, and significant damage to brand reputation.

The economic rationale for focusing on churn is compelling and well-documented. The cost of acquiring a new customer is substantially higher—estimated to be between five and twenty-five times more—than the cost of retaining an existing one. Furthermore, research has demonstrated that a mere 5% increase in customer retention rates can amplify profits by a remarkable 25% to 95%. This stark financial reality underscores the strategic importance of minimizing churn and makes its prediction and prevention a high-value, mission-critical activity for any organization.

The telecommunications sector serves as an archetypal case study for the challenges of customer churn. Characterized by intense competition and a high degree of market saturation, the industry experiences annual churn rates that can range from 15% to 25%. In this environment, customers have numerous service providers to choose from and can switch with relative ease, making customer loyalty a fragile and valuable commodity. Consequently, the ability to anticipate and mitigate churn is not just an advantage but a fundamental necessity for survival and growth.

### **1.2 The Solution: A Data-Driven Approach to Retention**

This project addresses the business challenge of customer churn by developing a robust, data-driven solution. The primary objective is to construct and evaluate a machine learning model capable of accurately predicting the probability of a customer churning. By leveraging historical customer data, the model can identify individuals who are at a high risk of leaving. This capability allows a business to transition from a reactive retention strategy, where interventions occur only after a customer has already decided to leave, to a proactive one, where targeted efforts can be deployed to retain at-risk customers before they churn.

A secondary, yet equally crucial, objective is to delve into the "why" behind customer attrition. Beyond simple prediction, this project will employ model interpretability techniques to identify and rank the key drivers and factors that most significantly influence a customer's decision to churn. By understanding which features—such as contract type, monthly charges, or tenure—are most predictive, a business can gain actionable intelligence. This intelligence is the foundation for developing highly targeted, effective, and resource-efficient retention campaigns, moving beyond generic offers to address the specific pain points of different customer segments.

### **1.3 Structuring for Success: A Professional Project Repository**

A hallmark of a professional data scientist is the ability to produce work that is not only accurate but also organized, reproducible, and maintainable. A well-structured project repository is fundamental to achieving these goals, signaling to collaborators and potential employers an understanding of best practices that extend beyond mere code execution.

For this project, we will adopt a standardized directory layout inspired by the widely-used cookiecutter-data-science template. This structure provides a logical and intuitive framework for managing the various components of a data science project, from raw data to final models and reports. The separation of concerns inherent in this structure—for instance, distinguishing between immutable raw data and processed data, or between exploratory notebooks and production-ready source code—is a cornerstone of professional data science and software development. It ensures that the project is easy to navigate, understand, and build upon, which is essential for both solo and team-based work. The chosen directory structure is detailed in the table below.

**Table 1: Project Directory Structure**

| Directory | Purpose |
| :---- | :---- |
| README.md | The top-level README for developers using this project. Contains a project overview, setup instructions, and navigation guide. |
| data/ | Contains all data for the project. |
| ├── raw/ | The original, immutable data dump. Data here should never be modified. |
| ├── processed/ | The final, canonical datasets for modeling after cleaning, transformation, and feature engineering. |
| notebooks/ | Jupyter notebooks for exploration, analysis, and prototyping. Notebooks are named sequentially (e.g., 1.0-data-exploration.ipynb). |
| reports/ | Generated analysis, such as this report, presentations, or dashboards. |
| ├── figures/ | Generated graphics and figures to be used in reporting. |
| models/ | Trained and serialized models, model predictions, or model summaries. |
| src/ | Source code for use in this project. This includes helper functions, data processing pipelines, and modeling scripts. |

## **II. Understanding the Data: An Initial Exploration**

### **2.1 Dataset Profile: IBM Telco Customer Churn**

The foundation of this project is the "Telco Customer Churn" dataset, a well-known and publicly available resource curated by IBM Sample Data Sets. It is a canonical dataset for classification tasks, particularly churn prediction, due to its clean structure and rich feature set. The dataset contains 7,043 rows, each representing a unique customer, and 21 columns, or features, that describe the customer's demographics, account information, subscribed services, and churn status. The target variable for our predictive task is the Churn column, which indicates whether the customer left the company within the last month. A comprehensive breakdown of each feature is provided in the data dictionary below.

**Table 2: Data Dictionary for Telco Customer Churn Dataset**

| Column Name | Data Type | Description | Role |
| :---- | :---- | :---- | :---- |
| customerID | object | A unique identifier for each customer. | Identifier |
| gender | object | The customer's gender (Male, Female). | Predictor |
| SeniorCitizen | int64 | Whether the customer is a senior citizen (1, 0). | Predictor |
| Partner | object | Whether the customer has a partner (Yes, No). | Predictor |
| Dependents | object | Whether the customer has dependents (Yes, No). | Predictor |
| tenure | int64 | The number of months the customer has been with the company. | Predictor |
| PhoneService | object | Whether the customer has a phone service (Yes, No). | Predictor |
| MultipleLines | object | Whether the customer has multiple lines (Yes, No, No phone service). | Predictor |
| InternetService | object | Customer's internet service provider (DSL, Fiber optic, No). | Predictor |
| OnlineSecurity | object | Whether the customer has online security (Yes, No, No internet service). | Predictor |
| OnlineBackup | object | Whether the customer has online backup (Yes, No, No internet service). | Predictor |
| DeviceProtection | object | Whether the customer has device protection (Yes, No, No internet service). | Predictor |
| TechSupport | object | Whether the customer has tech support (Yes, No, No internet service). | Predictor |
| StreamingTV | object | Whether the customer has streaming TV (Yes, No, No internet service). | Predictor |
| StreamingMovies | object | Whether the customer has streaming movies (Yes, No, No internet service). | Predictor |
| Contract | object | The customer's contract term (Month-to-month, One year, Two year). | Predictor |
| PaperlessBilling | object | Whether the customer has paperless billing (Yes, No). | Predictor |
| PaymentMethod | object | The customer's payment method (e.g., Electronic check, Mailed check). | Predictor |
| MonthlyCharges | float64 | The amount charged to the customer monthly. | Predictor |
| TotalCharges | object | The total amount charged to the customer over their lifetime. | Predictor |
| Churn | object | Whether the customer has churned (Yes, No). | Target |

### **2.2 First Look: Data Loading, Integrity, and Initial Cleaning**

The first step in any data analysis workflow is to load the data and perform an initial integrity check. This involves examining data types, checking for missing values, and identifying any immediate inconsistencies that require correction. The TotalCharges column was identified as an object type with 11 missing values. Investigation revealed these missing values corresponded to customers with 0 tenure, meaning they were new customers who had not yet been billed. The missing values were correctly imputed with 0\.

### **2.3 The Target Variable: Churn Distribution**

An analysis of the target variable, Churn, revealed a class imbalance. Approximately **73.5%** of customers were retained (No Churn), while **26.5%** churned. This imbalance necessitates the use of specific evaluation metrics (Precision, Recall, F1-Score, ROC AUC) and resampling techniques (SMOTE) to prevent the model from becoming biased towards the majority class.

## **III. Exploratory Data Analysis (EDA): Uncovering Patterns and Insights**

A systematic visual analysis was conducted to uncover the relationships between customer attributes and churn.

* **Demographics:** Senior citizens and customers without partners or dependents showed a significantly higher propensity to churn.  
* **Services:** Customers with **Fiber optic** internet churned at a much higher rate. A critical finding was that customers who did **not** subscribe to value-added services like OnlineSecurity, OnlineBackup, DeviceProtection, and TechSupport were far more likely to churn.  
* **Account Information:** **Month-to-month contracts** were identified as the single strongest indicator of churn. Customers paying by **Electronic check** also had a higher churn rate.  
* **Numerical Features:** Churn was heavily concentrated among customers with **low tenure** and **high monthly charges**.

A clear profile of a high-risk customer emerged: a new customer on a month-to-month contract, likely paying higher fees for fiber optic internet, without subscribing to protective add-on services.

## **IV. Data Preprocessing and Feature Engineering**

The data was prepared for modeling through a series of steps:

1. **Encoding:** Categorical features were converted into a numerical format using one-hot encoding.  
2. **Scaling:** Numerical features were standardized using StandardScaler to ensure they were on a comparable scale.  
3. **Splitting:** The data was split into an 80% training set and a 20% test set.  
4. **Resampling:** To address the class imbalance, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied *only* to the training data. This created a balanced dataset for model training while preserving the real-world distribution in the test set for a valid evaluation.

## **V. Predictive Modeling: Building and Comparing Classifiers**

Three classification models were trained on the preprocessed, resampled data:

1. **Logistic Regression:** A simple, interpretable baseline model.  
2. **Random Forest Classifier:** A powerful ensemble model that combines multiple decision trees to reduce overfitting.  
3. **Gradient Boosting Classifier:** A state-of-the-art ensemble model that builds trees sequentially, with each new tree correcting the errors of the previous ones.

## **VI. Model Evaluation and Selection**

The models were evaluated on the unseen test set using a comprehensive suite of metrics.

**Table 3: Model Performance Comparison**

|  | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Logistic Regression** | 0.748 | 0.518 | 0.782 | 0.623 | 0.839 |
| **Random Forest** | 0.785 | 0.598 | 0.626 | 0.612 | 0.835 |
| **Gradient Boosting** | 0.793 | 0.610 | 0.685 | 0.645 | 0.851 |

The **Gradient Boosting Classifier** was selected as the champion model. It demonstrated the best overall performance, achieving the highest Accuracy, F1-Score, and ROC AUC score (0.851), indicating a strong balance between correctly identifying churners (Recall) and not misclassifying loyal customers (Precision).

## **VII. Interpreting the Results: From Model to Strategy**

### **7.1 Identifying Key Churn Drivers: Feature Importance**

The feature importance scores from the trained Gradient Boosting model confirmed the insights from the EDA. The most influential factors in predicting churn were:

1. **Contract:** Month-to-month  
2. **Tenure**  
3. **Total Charges**  
4. **Internet Service:** Fiber optic  
5. **Monthly Charges**

### **7.2 Actionable Business Recommendations**

Based on these data-driven insights, four key business recommendations were formulated:

1. **Incentivize Long-Term Contracts:** Launch targeted campaigns to encourage month-to-month customers to upgrade to one or two-year plans by offering discounts or perks.  
2. **Enhance Early-Stage Customer Onboarding:** Implement a "First 90 Days" program with proactive check-ins and support to build loyalty with new, high-risk customers.  
3. **Bundle and Promote Protective Add-On Services:** Create discounted bundles for services like Tech Support and Online Security to increase customer investment and perceived value.  
4. **Review Fiber Optic Service and Pricing:** Survey fiber customers to identify pain points (price, reliability) and explore options like new pricing tiers or service improvements.

### **7.3 Conclusion and Future Work**

This project successfully developed a high-performing machine learning model to predict customer churn and translated its findings into actionable business strategies. Future work could include deploying the model as a real-time API, performing hyperparameter tuning to further enhance performance, and incorporating additional data sources for a more holistic customer view.