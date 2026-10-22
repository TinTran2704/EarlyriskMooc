# 📓 Notebook

Thư mục chứa toàn bộ **Jupyter Notebook** để tái lập (reproduce) thí nghiệm: từ xây dựng dữ liệu, huấn luyện mô hình đề xuất, đến chạy các baseline và tổng hợp kết quả.

> This folder contains all **Jupyter Notebooks** to reproduce the experiments: from data construction and training the proposed model, to running baselines and aggregating results.

---

## 📁 Cấu trúc thư mục / Folder structure

```
Notebook/
├── 01_build_model_base_data.ipynb               # Dựng dữ liệu nền / Build base data
├── 02_graphgrued_build_weekly_active_dataset.ipynb
├── 03_graphgrued_relabel_ordered5.ipynb
├── 04_graphgrued_graph_impute_ordered5_fixed.ipynb
├── 05_graphgrued_ordered5_prepare_windowed.ipynb
├── 06_graphgrued_ordered5_window_experiments.ipynb
├── 07_graphgrued_ordered5_window_report.ipynb   # Báo cáo mô hình đề xuất / Proposed-model report
│
├── utilities.ipynb                              # Hàm dùng chung / Shared helper functions
│
└── baseline/                                    # Các mô hình so sánh / Baseline models
    ├── 02_baseline_fei_yeung_2015.ipynb
    ├── 03_baseline_alamri_2019.ipynb
    ├── 04_baseline_dass_2021.ipynb
    ├── 05_baseline_xing_du_2019.ipynb
    ├── 06_baseline_basnet_2022.ipynb
    ├── 07_baseline_aouarib_2025.ipynb
    └── 08_compare_all_baselines.ipynb           # Tổng hợp / Aggregate
```

---

## ▶️ Thứ tự chạy / Execution order

**Bước 1 — Chạy notebook ở thư mục ngoài theo thứ tự `01 → 07`.**
Run the outer notebooks **in order from `01` to `07`** first.

**Bước 2 — Sau đó chạy các notebook trong `baseline/` theo thứ tự.**
Then run the notebooks inside `baseline/` **in order**.

> ✅ Kết quả cuối cùng là **một file CSV** tổng hợp toàn bộ chỉ số đánh giá.
> The final output is **a single CSV file** summarizing all evaluation metrics.

---

## 📌 Ghi chú / Notes

- `utilities.ipynb` chứa các hàm dùng chung; đảm bảo nó nằm cùng thư mục để các notebook khác import được.
  *`utilities.ipynb` holds shared functions; keep it in place so other notebooks can import it.*
- Hãy chạy lần lượt **từng cell theo đúng thứ tự** trong mỗi notebook, không bỏ qua bước.
  *Run cells sequentially within each notebook; do not skip steps.*
- Notebook đánh số nối tiếp nhau theo luồng dữ liệu — đầu ra bước trước là đầu vào bước sau.
  *Notebooks are numbered to follow the data flow — each step's output feeds the next.*
