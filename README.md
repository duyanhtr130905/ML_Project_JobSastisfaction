# Dự đoán Mức độ Hài lòng Nhân viên - IBM HR Analytics

## Thông tin nhóm
- **Thành viên 1**: [Họ tên] - [MSSV] - [Email]
  - Công việc: Data Preprocessing, EDA, Clustering, Report Writing
- **Thành viên 2**: [Họ tên] - [MSSV] - [Email]
  - Công việc: Dimensionality Reduction, Regression Models, Classification Models

## Mô tả đề tài
Xây dựng mô hình học máy dự đoán mức độ hài lòng của nhân viên dựa trên các yếu tố cá nhân và công việc.

## Dataset
- **Tên**: IBM HR Analytics Employee Attrition & Performance
- **Nguồn**: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **Kích thước**: 1470 samples, 35 features
- **Target**: JobSatisfaction (1-5)

## Cài đặt môi trường
```bash
# Clone repository
git clone [your-repo-url]
cd ML_Project_JobSatisfaction

# Cài đặt thư viện
pip install -r requirements.txt
```

## Cấu trúc dữ liệu
Tải dataset từ Kaggle và đặt vào thư mục `data/raw/`

## Chạy thực nghiệm

### Cách 1: Chạy từng notebook
```bash
# Mở Jupyter Notebook
jupyter notebook

# Chạy lần lượt các notebook trong thư mục notebooks/
# theo thứ tự 01 -> 06
```

### Cách 2: Chạy script tổng hợp
```bash
python run_experiments.py
```

## Kịch bản thực nghiệm

### Experiment 1: Regression trên dữ liệu gốc
- Models: Linear Regression, Ridge, Lasso, KNN, MLP
- Train/Test splits: 80/20, 70/30, 60/40
- Metrics: RMSE, MAE, R²

### Experiment 2: Regression trên dữ liệu giảm chiều
- Dimensionality Reduction: PCA (giảm xuống 1/3 số chiều)
- Same models và splits như Exp 1

### Experiment 3: Classification
- Transform JobSatisfaction → 3 classes (Low, Medium, High)
- Models: Naive Bayes, Logistic Regression, Decision Tree, SVM
- Train/Test splits: 80/20, 70/30, 60/40

### Experiment 4: Clustering
- Methods: K-Means, GMM
- K values: 2-10 (chọn optimal K)
- Evaluation: Silhouette Score, Davies-Bouldin Index

## Kết quả

### Hồi quy
- **Best Model**: Ridge Regression
- **Best R²**: 0.XX (trên dữ liệu gốc, split 80/20)
- **Best RMSE**: X.XX

### Phân loại
- **Best Model**: Logistic Regression
- **Best Accuracy**: XX%
- **Best F1-Score**: 0.XX

## Báo cáo và Slides
- Báo cáo PDF: `reports/ML_Project_Report.pdf`
- Slides: `reports/ML_Project_Slides.pptx`

## Tài liệu tham khảo
1. Scikit-learn Documentation
2. IBM HR Analytics Dataset Documentation
3. [Thêm tài liệu khác...]