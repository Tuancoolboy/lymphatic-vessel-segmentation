# Phân đoạn mạch bạch huyết với CTO-Net

Dự án này cung cấp một quy trình bán giám sát (semi-supervised) để phân đoạn các mạch bạch huyết trong video. Kiến trúc chính được sử dụng là **CTO-Net**, một mô hình tùy chỉnh với backbone **Res2Net-50**. Quy trình này sử dụng phương pháp **Mean Teacher** để học bán giám sát. Dự án cũng hỗ trợ mô hình **UNet++** và một mô hình thử nghiệm là **CTO Stitch-ViT**.

## Mục lục
- [Tính năng nổi bật](#tính-năng-nổi-bật)
- [Các mô hình](#các-mô-hình)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
  - [Quy trình Huấn luyện](#quy-trình-huấn-luyện)
  - [Chạy với mô hình Stitch-ViT](#chạy-với-mô-hình-stitch-vit)
- [Công cụ hỗ trợ](#công-cụ-hỗ-trợ-tools)
- [Ứng dụng GUI](#ứng-dụng-gui)
- [Kết quả](#kết-quả)
- [Chi tiết kỹ thuật](#chi-tiết-kỹ-thuật)

## Tính năng nổi bật

*   **Kiến trúc chính (CTO-Net):** Một mô hình phân đoạn hiệu suất cao với backbone **Res2Net-50**, được thiết kế để trích xuất đặc trưng chi tiết.
*   **Học bán giám sát:** Tận dụng dữ liệu video chưa gán nhãn thông qua phương pháp **Mean Teacher** để cải thiện độ chính xác và giảm công sức gán nhãn.
*   **Hàm mất mát nhận biết đường viền:** Kết hợp Dice Loss và Boundary Loss để phát hiện các cạnh mạch máu sắc nét và chính xác.
*   **Quy trình huấn luyện 2 giai đoạn:**
    1.  **Stage 1:** Huấn luyện mô hình cơ sở (baseline) trên dữ liệu có nhãn.
    2.  **Stage 2:** Tinh chỉnh mô hình bằng Mean Teacher với cả dữ liệu có nhãn và không nhãn.
*   **Cấu hình linh hoạt:** Dễ dàng tùy chỉnh tham số cho từng giai đoạn thông qua các file JSON chuyên dụng.
*   **GUI phân tích:** Công cụ đồ họa trực quan để xem dự đoán và đo đường kính mạch.

## Các mô hình

Dự án này hỗ trợ ba mô hình:

1.  **CTO-Net (Mặc định):** Mô hình chính của dự án. Nó sử dụng backbone **Res2Net-50** và một kiến trúc tùy chỉnh được thiết kế để phân đoạn mạch máu. Cấu hình cho mô hình này là `cto`.
2.  **CTO Stitch-ViT (Thử nghiệm):** Một phiên bản nâng cao của CTO-Net, tích hợp các khối Vision Transformer (ViT) với cơ chế "ghép nối" (stitching). Mô hình này nhằm mục đích nắm bắt các đặc trưng ngữ cảnh toàn cục hơn. Cấu hình cho mô hình này là `cto_stitchvit`.
3.  **UNet++:** Một kiến trúc nổi tiếng cho phân đoạn ảnh y tế, có sẵn dưới dạng một lựa chọn thay thế. Nó có thể được cấu hình bằng cách đặt `name` của mô hình thành `unetpp` trong file config JSON.

## Cấu trúc dự án

```text
.
├── app.py                      # File chạy ứng dụng GUI
├── config_stage1.json          # Cấu hình cho Stage 1 (CTO-Net)
├── config_stage2.json          # Cấu hình cho Stage 2 (CTO-Net)
├── config_stage1_stitchvit.json # Cấu hình cho Stage 1 (Stitch-ViT)
├── config_stage2_stitchvit.json # Cấu hình cho Stage 2 (Stitch-ViT)
├── requirements.txt
├── data/
│   ├── annotated/              # Ảnh và file JSON đã gán nhãn
│   │   ├── Human/
│   │   └── Rat/
│   ├── masks/                  # Ảnh mặt nạ (mask) đã được tạo ra
│   │   ├── Human/
│   │   └── Rat/
│   ├── frames/                 # Các khung hình được trích xuất từ video
│   │   ├── Human/
│   │   └── Rat/
│   └── video/                  # Video thô chưa gán nhãn
│       ├── Human/
│       └── Rat/
├── logs/                       # Nhật ký huấn luyện, biểu đồ và ảnh dự đoán
│   ├── Human/
│   └── Rat/
├── models/                     # Nơi lưu các checkpoint của mô hình (.pth)
│   └── checkpoints/
│       ├── Human/
│       └── Rat/
└── src/                        # Mã nguồn
    └── models/
        ├── cto/                # Mã nguồn cho CTO-Net chính
        └── cto_stitchvit/      # Mã nguồn cho biến thể Stitch-ViT
```

## Cài đặt

1.  **Tạo và kích hoạt môi trường ảo:**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Cấu hình loại dữ liệu:**
    Mở các file cấu hình bạn dự định sử dụng (ví dụ: `config_stage1.json`). Đặt trường `"type"` thành `"Human"` hoặc `"Rat"` để xác định bộ dữ liệu của bạn. Hệ thống sẽ tự động sử dụng các thư mục con tương ứng.

    **Ví dụ (`config_stage1.json`):**
    ```json
    {
        "type": "Human",
        ...
    }
    ```

## Hướng dẫn sử dụng

### Chuẩn bị dữ liệu

1.  **Dữ liệu có nhãn:** Đặt ảnh và file annotation `.json` của bạn vào `data/annotated/<type>/` (ví dụ: `data/annotated/Human/`).
2.  **Chuyển đổi Annotation sang Mask:** Tạo ảnh mặt nạ nhị phân để huấn luyện.
    ```bash
    python -m tools.scripts.convert_json_to_mask --input data/annotated --output data/masks
    ```
3.  **Dữ liệu chưa nhãn:** Đặt các video thô của bạn vào `data/video/<type>/`.
4.  **Trích xuất khung hình từ Video:** Chuẩn bị các khung hình cho việc học bán giám sát.
    ```bash
    python -m tools.scripts.extract_frames --video_dir data/video --output_dir data/frames --fps 1
    ```

### Quy trình Huấn luyện

Cách được khuyến nghị để huấn luyện là chạy toàn bộ quy trình, thực thi tuần tự Giai đoạn 1 và Giai đoạn 2 bằng mô hình **CTO-Net** mặc định.

**Chạy toàn bộ quy trình (CTO-Net):**
```bash
python -m src.main all --visualize
```
Lệnh này sử dụng `config_stage1.json` và `config_stage2.json` làm mặc định.

**Chạy riêng lẻ từng giai đoạn (CTO-Net):**
Bạn cũng có thể chạy riêng từng giai đoạn.

*   **Giai đoạn 1: Huấn luyện Baseline**
    ```bash
    python -m src.main baseline --config config_stage1.json
    ```
*   **Giai đoạn 2: Huấn luyện Final (Mean Teacher)**
    ```bash
    python -m src.main final --config config_stage2.json
    ```

### Chạy với mô hình Stitch-ViT

Để sử dụng mô hình **CTO Stitch-ViT**, bạn phải chỉ định file cấu hình của nó bằng cách sử dụng cờ `--config`.

**Toàn bộ quy trình (Stitch-ViT):**
```bash
# Đầu tiên, chạy Giai đoạn 1 với file config của nó
python -m src.main baseline --config config_stage1_stitchvit.json --visualize

# Sau đó, chạy Giai đoạn 2 với file config tương ứng (nó sẽ sử dụng trọng số từ Giai đoạn 1)
python -m src.main final --config config_stage2_stitchvit.json --visualize
```

### Các cờ (flag) bổ sung

*   `--config <path>`: Sử dụng một file cấu hình tùy chỉnh.
*   `--small-test`: Chạy trên một tập dữ liệu con nhỏ để gỡ lỗi.
*   `--visualize`: Tạo và lưu biểu đồ dự đoán sau khi huấn luyện.
*   `--early-stop-patience <int>`: Ghi đè giá trị kiên nhẫn của early stopping từ file cấu hình.

## Công cụ hỗ trợ (Tools)

Các script hữu ích được đặt trong `tools/scripts/`.

*   **So sánh các mô hình:**
    Tạo ra một so sánh trực quan về các dự đoán từ hai mô hình khác nhau.
    ```bash
    python -m tools.scripts.compare_models --log-dir1 <đường_dẫn_tới_log_model1> --log-dir2 <đường_dẫn_tới_log_model2>
    ```

*   **Vẽ biểu đồ huấn luyện:**
    Tạo các biểu đồ loss và metric từ một thư mục log huấn luyện.
    ```bash
    python -m tools.scripts.plot_training_curves --log-dir <đường_dẫn_tới_thư_mục_log>
    ```

*   **Trực quan hóa dự đoán:**
    Tải một mô hình đã huấn luyện để tạo và lưu hình ảnh dự đoán trên một tập kiểm tra.
    ```bash
    python -m tools.scripts.visualize_predictions --log-dir <đường_dẫn_tới_thư_mục_log>
    ```

## Ứng dụng GUI

Dự án bao gồm một giao diện người dùng đồ họa để dự đoán và phân tích tương tác.

**Khởi chạy GUI:**
```bash
python app.py
```

## Kết quả

Sau khi chạy một quy trình huấn luyện, các kết quả đầu ra được lưu trong thư mục `logs/` và `models/`, được sắp xếp theo `type` và `experiment_name` từ file cấu hình của bạn.

Với một thí nghiệm có tên `"cto"` thuộc loại `"Human"`, bạn sẽ tìm thấy:
*   **Model Checkpoints:** `models/checkpoints/Human/cto/baseline.pth` và `models/checkpoints/Human/cto/final.pth`.
*   **Nhật ký huấn luyện & Hình ảnh:** `logs/Human/cto/`. Thư mục này chứa:
    *   `_detailed_curves.png`: Biểu đồ Dice và loss.
    *   `_loss_curve.png`: Biểu đồ tổng hợp loss.
    *   `baseline_predictions.png`: Kết quả trực quan từ mô hình baseline.
    *   `final_predictions.png`: Kết quả trực quan từ mô hình Mean Teacher cuối cùng.
    *   `training.log`: Một file văn bản chứa nhật ký chi tiết.

## Chi tiết kỹ thuật

*   **Mô hình cốt lõi:** CTO-Net với backbone Res2Net-50.
*   **Chiến lược bán giám sát:** Mean Teacher, trong đó trọng số của mô hình Teacher là Trung bình Động Hàm mũ (Exponential Moving Average - EMA) của trọng số của Student. Student học từ cả dữ liệu có nhãn (supervised loss) và dự đoán của Teacher trên dữ liệu không nhãn (consistency loss).
*   **Hàm mất mát:** Một sự kết hợp của Binary Cross-Entropy (BCE), Dice Loss, và một Boundary Loss tùy chỉnh.
