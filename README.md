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

---

# 🌐 Hệ thống Dịch thuật Anh-Việt cho Tài liệu Khoa học

## Tổng quan

Hệ thống dịch thuật chuyên sâu sử dụng công nghệ tiên tiến:

- **MarianMT**: Mô hình dịch máy nguồn mở chạy cục bộ (Edge AI)
- **RAG (Retrieval-Augmented Generation)**: Bộ nhớ dịch thông minh
- **Knowledge Graph**: Chuẩn hóa thuật ngữ chuyên ngành
- **Drupal CMS**: Nền tảng quản lý nội dung
- **Apache Superset**: Phân tích hiệu suất theo thời gian thực
- **Mobile App**: Ứng dụng đa nền tảng (iOS & Android)

## Cấu trúc dự án

```
translation_system/
├── core/               # MarianMT translation engine
├── rag/                # Translation memory with RAG
├── knowledge_graph/    # Terminology standardization
├── api/                # FastAPI REST backend
├── drupal/             # Drupal CMS integration
├── mobile/             # React Native mobile app
├── analytics/          # Apache Superset analytics
├── config/             # Configuration files
├── examples/           # Usage examples
└── README.md           # Detailed documentation
```

## Tính năng chính

### 1. Động cơ dịch thuật (MarianMT)
- Dịch Anh-Việt chuyên ngành khoa học
- Chạy cục bộ (Edge AI) - không cần kết nối internet
- Hỗ trợ dịch đơn lẻ và hàng loạt

### 2. Bộ nhớ dịch thông minh (RAG)
- Lưu trữ và tái sử dụng bản dịch
- Tìm kiếm bản dịch tương tự
- Đảm bảo tính nhất quán trong toàn bộ tài liệu

### 3. Knowledge Graph
- Quản lý thuật ngữ chuyên ngành
- Đồng bộ hóa thuật ngữ trong toàn tài liệu
- Hỗ trợ nhiều lĩnh vực khoa học

### 4. API Backend
- REST API với FastAPI
- Tài liệu tự động (Swagger UI)
- Xác thực và bảo mật

### 5. Drupal CMS
- Quản lý nội dung dịch thuật
- Giao diện người dùng thân thiện
- Tích hợp với API backend

### 6. Apache Superset
- Dashboard phân tích thời gian thực
- Theo dõi hiệu suất dịch thuật
- Thống kê sử dụng

### 7. Mobile App
- Ứng dụng đa nền tảng (iOS & Android)
- Dịch thuật ngoại tuyến
- Lịch sử dịch thuật

## Cài đặt nhanh

### Sử dụng Docker (Khuyến nghị)

```bash
cd translation_system
docker-compose up -d
```

### Cài đặt thủ công

```bash
# Cài đặt dependencies
cd translation_system
pip install -r requirements.txt

# Khởi động API server
cd api
python main.py
```

## Sử dụng

### API Documentation
- Truy cập: http://localhost:8000/docs
- Swagger UI để test các endpoint

### Ví dụ sử dụng

```python
from core.translator import TranslationEngine
from rag.translation_memory import TranslationMemory, RAGTranslationEngine
from knowledge_graph.terminology import create_default_terminology

# Khởi tạo
engine = TranslationEngine()
memory = TranslationMemory()
kg = create_default_terminology()

# Dịch văn bản
result = engine.translate("Machine learning is powerful")
print(result['target'])  # Học máy rất mạnh mẽ
```

### Xem thêm ví dụ

```bash
cd translation_system/examples
python usage_examples.py
```

## Tài liệu chi tiết

Xem [translation_system/README.md](translation_system/README.md) để biết thêm chi tiết về:
- Kiến trúc hệ thống
- Hướng dẫn sử dụng chi tiết
- API endpoints
- Cấu hình
- Deployment
- Performance tuning

## Yêu cầu hệ thống

- Python 3.8+
- 4GB+ RAM
- Docker & Docker Compose (khuyến nghị)
- Node.js 16+ (cho mobile app)

## Liên hệ

Để biết thêm thông tin về hệ thống dịch thuật, vui lòng tham khảo tài liệu trong thư mục `translation_system/`.