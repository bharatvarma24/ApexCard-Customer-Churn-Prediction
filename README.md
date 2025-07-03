# ApexCard-Customer-Churn-Prediction
# ApexCard-Customer-Churn-Prediction

## Project Overview

This project focuses on developing a robust machine learning solution to predict customer churn for **ApexCard**, a leading fictional credit card services provider. The primary objective is to identify customers at high risk of churning, understand the underlying drivers of this behavior, and provide actionable, data-driven recommendations to improve customer retention and lifetime value.

Customer churn is a critical business challenge in the financial sector, directly impacting revenue and profitability. By proactively identifying at-risk customers, ApexCard can implement targeted retention strategies, significantly reducing customer attrition costs and enhancing overall customer satisfaction.

## Methodology & Technical Approach

This project follows a comprehensive data science lifecycle, leveraging SQL and Python for data manipulation, feature engineering, model development, and evaluation.

### 1. Data Simulation

* **Purpose:** To create a realistic, large-scale dataset mimicking a credit card company's operational data, as real financial data is sensitive and proprietary. This demonstrates the ability to work with complex, interconnected datasets.
* **Technical Details:**
    * Utilized **Python's `Faker` library** to generate synthetic customer demographics (names, emails, dates of birth, regions).
    * Generated interconnected data for `Customers`, `Accounts`, `Transactions`, `Payments`, and `CustomerServiceInteractions` tables.
    * Populated an **in-memory SQLite database** using `pandas.to_sql()` for efficient data storage and SQL querying within the Python environment (Google Colab).
    * Simulated a realistic churn rate (approximately 2.9%) by explicitly closing a small percentage of accounts.

### 2. SQL-based Feature Engineering

* **Purpose:** To transform raw, granular transactional, payment, and interaction data into aggregated, predictive features at the account level. All features were calculated up to a defined `OBSERVATION_DATE` (2024-06-30) to prevent data leakage.
* **Technical Details:**
    * Defined the **target variable `is_churned`** based on explicit account closures by the observation date.
    * Aggregated `Transactions` data to create features such as:
        * `total_transactions_overall`, `total_spending_overall`, `avg_transaction_amount_overall`
        * Recent activity metrics (`total_transactions_last_3_months`, `total_spending_last_3_months`, `avg_transaction_amount_last_3_months`)
        * `unique_merchant_categories_last_3_months` (a key indicator of engagement diversity).
    * Aggregated `Payments` data to derive features like:
        * `total_payments_overall`, `total_amount_paid_overall`, `avg_payment_amount_overall`
        * Recent payment activity (`total_payments_last_3_months`, `total_amount_paid_last_3_months`, `avg_payment_amount_last_3_months`).
    * Aggregated `CustomerServiceInteractions` data, focusing on:
        * `total_interactions_overall`, `unresolved_interactions_overall`
        * Recent interaction patterns (`total_interactions_last_3_months`, `unresolved_interactions_last_3_months`)
        * Specific high-impact issue types (`account_closure_requests_last_3_months`, `billing_disputes_last_3_months`).

### 3. Python-based Data Preprocessing

* **Purpose:** To prepare the engineered features for machine learning model training.
* **Technical Details:**
    * **Time-Based Feature Creation:** Derived crucial numerical features from date columns (e.g., `account_tenure_days`, `days_since_last_transaction`, `days_since_last_payment`, `days_since_last_interaction`, `customer_age_at_obs`). Handled `NaN` values by imputing with a `VERY_OLD_DATE` to signify prolonged inactivity.
    * **Missing Value Handling:** Systematically identified and imputed remaining `NaN` values in numerical columns (e.g., average transaction amounts for accounts with no transactions were filled with 0).
    * **Categorical Encoding:** Applied **One-Hot Encoding** using `pandas.get_dummies()` to convert nominal categorical features (e.g., `region`, `account_type`) into a numerical format suitable for ML models, while avoiding multicollinearity (`drop_first=True`).
    * **Feature Scaling:** Employed **`StandardScaler` from Scikit-learn** to normalize numerical features, ensuring they are on a comparable scale and preventing features with larger magnitudes from dominating model training. Binary (one-hot encoded) features were intentionally excluded from scaling.
    * **Data Splitting:** Divided the preprocessed data into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to maintain the original churn class proportion, crucial for imbalanced datasets.

### 4. Machine Learning Model Development & Evaluation

* **Purpose:** To train predictive models and rigorously assess their ability to identify churners on unseen data.
* **Technical Details:**
    * **Model Selection:** Trained two classification models:
        * **Logistic Regression:** As a linear baseline, initialized with `class_weight='balanced'` to address class imbalance.
        * **Random Forest Classifier:** A powerful ensemble tree-based model, also initialized with `class_weight='balanced'` for robust handling of imbalance and non-linear relationships.
    * **Evaluation Metrics:** Assessed model performance using:
        * **Classification Report:** Providing Precision, Recall, and F1-Score for both churn and non-churn classes.
        * **ROC AUC Score:** A robust metric for imbalanced classification, measuring the model's ability to distinguish between classes.
        * **Confusion Matrix:** Visualizing True Positives, True Negatives, False Positives, and False Negatives.
    * **Visualizations:** Generated Confusion Matrix heatmaps and ROC Curves for both models to visually interpret performance.

### Key Findings

After training and evaluating both models, the **Random Forest Classifier** emerged as the superior choice for ApexCard's churn prediction:

* **Superior Discriminatory Power:** Random Forest achieved an **outstanding ROC AUC Score of 0.9837**, significantly outperforming Logistic Regression (0.9433). This indicates its excellent ability to rank customers by their likelihood of churn.
* **High Precision for Churn:** The Random Forest model demonstrated a remarkable **Precision of 0.95 for the churn class**. This means that when the model identifies a customer as likely to churn, it is correct 95% of the time, leading to minimal "false alarms" (only 2 in the test set compared to 174 by Logistic Regression).
* **Balanced Performance:** While its Recall for churn (0.59) was lower than Logistic Regression (0.80), the dramatically higher precision and overall AUC make Random Forest more efficient for targeted retention efforts, minimizing wasted resources on customers who would not have churned.

**Top Churn Drivers (from Random Forest Model):**

The Random Forest model's feature importance analysis provided clear insights into the most influential factors driving churn for ApexCard customers:

1.  **Inactivity (e.g., `days_since_last_transaction`, `days_since_last_payment`):** The single most critical indicator. Longer periods without any card activity (transactions or payments) are the strongest predictors of churn.
2.  **Declining Recent Engagement (e.g., `total_transactions_last_3_months`, `total_spending_last_3_months`):** A significant reduction in transactional volume or spending amount in the recent three months is a strong precursor to churn.
3.  **Lack of Diverse Card Usage (`unique_merchant_categories_last_3_months`):** Customers who use their ApexCard across a narrow range of merchant categories are more prone to churn. Diverse usage signifies deeper integration of the card into their financial habits.
4.  **Customer Tenure (`account_tenure_days`):** While less impactful than recent activity, longer customer tenure generally correlates with lower churn risk.
5.  **Customer Service Interactions:** While overall interactions were less important in Random Forest than in Logistic Regression, specific types or unresolved issues could still be indicators (though not top-ranked in this model's importance).

### Actionable Business Recommendations for ApexCard

Based on these robust findings, ApexCard can implement the following data-driven strategies to proactively manage and reduce customer churn:

1.  **Implement Proactive Inactivity Alerts & Re-engagement Campaigns:**
    * **Strategy:** Establish automated systems to monitor `days_since_last_transaction` and `days_since_last_payment`. When these metrics exceed predefined thresholds (e.g., 30, 60, or 90 days), trigger immediate, personalized re-engagement campaigns.
    * **Tactics:**
        * **Personalized Offers:** Send targeted emails or in-app notifications with exclusive rewards, bonus points for the next transaction, or special interest rate offers to reactivate dormant accounts.
        * **Value Reminders:** Highlight unused card benefits, loyalty program points, or cashback opportunities.
        * **"We Miss You" Outreach:** A simple, empathetic message checking on satisfaction and offering assistance.

2.  **Develop Programs to Foster Diverse Card Usage:**
    * **Strategy:** Encourage customers to integrate ApexCard more deeply into their daily spending across various categories, as indicated by `unique_merchant_categories_last_3_months`.
    * **Tactics:**
        * **Gamification:** Introduce challenges or bonus rewards for spending in new merchant categories.
        * **Tiered Rewards:** Offer higher reward rates for customers who diversify their spending.
        * **Personalized Spending Insights:** Provide customers with dashboards showing their spending across categories and suggesting ways to maximize rewards by using ApexCard more broadly.

3.  **Monitor and Intervene on Declining Recent Activity:**
    * **Strategy:** Create a system to detect significant drops in `total_transactions_last_3_months` and `total_spending_last_3_months` compared to a customer's historical average.
    * **Tactics:**
        * **Early Warning System:** Flag accounts showing a sharp decline (e.g., 25% or more) in recent activity.
        * **Competitor Analysis:** Investigate if customers are shifting spending to competitors and tailor offers to regain share of wallet (e.g., balance transfer offers, competitive APRs).
        * **Customer Feedback Loop:** Proactively reach out to understand the reasons behind reduced activity.

4.  **Reinforce Loyalty for Long-Term Customers:**
    * **Strategy:** While long-tenure customers are less likely to churn, they are highly valuable. Continue to nurture these relationships.
    * **Tactics:**
        * **Anniversary Rewards:** Offer special bonuses or exclusive perks on account anniversaries.
        * **Tiered Benefits:** Introduce loyalty tiers that unlock increasing benefits based on tenure and engagement.
        * **Proactive Account Reviews:** Offer personalized financial health checks or credit limit reviews to long-standing customers.

5.  **Optimize Retention Resource Allocation:**
    * **Strategy:** Leverage the Random Forest model's high precision (0.95) to ensure retention efforts are highly efficient.
    * **Tactics:**
        * **Targeted Campaigns:** Focus marketing and customer service resources primarily on the customers the model flags with high churn probability, knowing that these predictions are highly reliable.
        * **Cost-Benefit Thresholding:** Use the model's predicted churn probabilities to set a dynamic threshold for intervention, balancing the cost of a false positive against the value of a prevented churn.

## Technologies Used

* **Python:**
    * `pandas` (Data manipulation and DataFrame operations)
    * `numpy` (Numerical operations)
    * `Faker` (Synthetic data generation)
    * `sqlite3` (In-memory SQL database interaction)
    * `scikit-learn` (Machine learning models, preprocessing, evaluation metrics)
    * `matplotlib` (Data visualization)
    * `seaborn` (Enhanced data visualization)
* **SQL:** For complex data aggregation and feature engineering.

## How to Replicate

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/ApexCard-Customer-Churn-Prediction.git](https://github.com/YourGitHubUsername/ApexCard-Customer-Churn-Prediction.git)
    cd ApexCard-Customer-Churn-Prediction
    ```
2.  **Open in Google Colab:** Upload the Python notebook (`.ipynb` file, which you can save from your current Colab session) to Google Colab.
3.  **Run Cells Sequentially:** Execute each code cell in the notebook from top to bottom. The notebook is structured to guide you through data simulation, feature engineering, preprocessing, model training, and evaluation.



---
