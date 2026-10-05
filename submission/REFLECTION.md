# Reflection — Lab 19

**Tên:** Nguyen Trung Kien
**Cohort:** A20-K4 (MSSV: 2A202602764)
**Path đã chạy:** both (Docker stack & Lite verification)

---

## Câu hỏi (≤ 200 chữ)

Trên 50 golden queries:
- **`exact` (96.7%):** BM25 và Hybrid cùng dẫn đầu do từ khóa kỹ thuật khớp chính xác trong tài liệu gốc.
- **`mixed` (100.0%):** Hybrid thắng tuyệt đối (vượt BM25 97.0% và Vector 98.5%) nhờ dung hòa giữa độ chính xác từ vựng và độ phủ ngữ nghĩa.
- **`paraphrase` (24%–33%):** Câu hỏi diễn đạt thuần Việt không chứa từ khóa gốc; do mô hình `bge-small-en` thiên tiếng Anh nên điểm giảm chung, chuyển sang `bge-m3` sẽ vượt trội.

**Khi nào KHÔNG dùng Hybrid:**
- **Chọn Pure BM25:** Khi tìm kiếm mã định danh (SKU, UUID, mã lỗi log, số hợp đồng) hoặc hệ thống bị giới hạn khắt khe về CPU/RAM/chi phí, không thể duy trì embedding inference.
- **Chọn Pure Vector:** Khi tìm kiếm đa phương thức (image/audio search), cross-lingual (truy vấn một ngôn ngữ, tìm tài liệu ngôn ngữ khác), hoặc kho dữ liệu phi cấu trúc hoàn toàn không có từ khóa cố định.

---

## Điều ngạc nhiên nhất khi làm lab này

Hiện tượng rò rỉ dữ liệu (data leakage) trong Target Encoding trên trường có cardinality cao (`session_id`) tạo ra lift ảo khổng lồ (Train AUC 0.999 vs Test AUC 0.522, gap tới 0.477), và post-filter làm sập recall về 0.00 khi độ chọn lọc hẹp (~3.8%).

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work với: _Không_
