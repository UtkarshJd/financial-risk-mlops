# 💰 AI-Powered Financial Risk Prediction System (AWS MLOps)

A complete end-to-end machine learning project to predict financial risk (e.g., loan default) using AWS services such as S3, Lambda, SageMaker, and Elastic Beanstalk with Dockerized Flask API.

---

## 🚩 Problem Statement

Traditional financial risk assessment methods are:
- Manual, slow, and error-prone
- Not scalable for large data
- Inefficient in meeting regulatory compliance

---

## 💡 Proposed Solution

An automated, AI-powered risk prediction system that:
- Ingests and processes financial data (CSV)
- Trains an ML model using AWS SageMaker
- Deploys a Flask API via Elastic Beanstalk for real-time prediction
- Uses Docker and AWS services for scalable MLOps

---

## ⚙️ Technologies Used

| Service            | Role                                             |
|--------------------|--------------------------------------------------|
| Amazon S3          | Store raw, processed data & model artifacts      |
| AWS Lambda         | Preprocess uploaded data                         |
| Amazon SageMaker   | Train the XGBoost ML model                       |
| Elastic Beanstalk  | Deploy the Flask API using Docker                |
| EC2 + Docker       | Host the containerized prediction API            |
| CloudWatch         | Logs, metrics, and monitoring                    |

---

## 🧱 Architecture Diagram

```mermaid
graph TD
    A[CSV Upload to S3] --> B[AWS Lambda - Preprocess]
    B --> C[S3 Processed Data]
    C --> D[SageMaker - Model Training]
    D --> E[S3 - Trained Model Storage]
    E --> F[Elastic Beanstalk - Flask API]
    F --> G[User Sends JSON Data for Prediction]
    G --> H[Flask API Returns Prediction]
