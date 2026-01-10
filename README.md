# Dự đoán Mức độ Hài lòng Nhân viên - IBM HR Analytics

## Thông tin nhóm
- **Thành viên 1**: Lê Huy Anh Dũng - 23001846 - 23001846@hus.edu.vn
  - Công việc: Data Preprocessing, EDA, Clustering, Report Writing
- **Thành viên 2**: Trần Duy Anh - 23001828 - 23001828@hus.edu.vn
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

```bash
# Mở Jupyter Notebook
jupyter notebook

# Chạy lần lượt các notebook trong thư mục notebooks/
# theo thứ tự 01 -> 06
```
## Kịch bản thực nghiệm

### Experiment 1: Phân loại (Classification)
- Target: JobSatisfaction (4 lớp: 1, 2, 3, 4)
- Model
Nhóm 1: Naive Bayes (GaussianNB)
Nhóm 2: Logistic Regression (Softmax)
Nhóm 3: SVM (RBF kernel)
- Dữ liệu:
Dữ liệu gốc (Original features)
Dữ liệu giảm chiều (PCA - 95% variance)
- Train/Test splits: 80/20, 70/30, 60/40
- Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Experiment 2: Hồi quy từ Hàm Quyết định
- Target mới:
Softmax probability từ Logistic Regression (cho lớp 4)
Decision function từ SVM (cho lớp 4)
- Models: Ridge Regression, MLP Regressor
- Dữ liệu:
Dữ liệu gốc (scaled)
Dữ liệu giảm chiều (PCA - 1/3 số chiều gốc)
- Train/Test splits: 80/20, 70/30, 60/40
- Metrics: RMSE, MAE, R², Residual Analysis

### Experiment 3: Dimensionality Reduction
- Methods: PCA, t-SNE
- PCA variants:
95% variance retained
1/3 số chiều gốc
- Visualization: 2D/3D plots with JobSatisfaction coloring

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