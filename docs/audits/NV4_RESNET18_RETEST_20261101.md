# Kiểm thử lại tích hợp ResNet18 sau commit NV3 `2e10e6a`

Ngày 01/11/2026, NV4 kiểm thử trong worktree tạm `../nv4_resnet_retest_20261101` tạo từ `origin/main` `9fc3d43`. Worktree ghép lần lượt HEAD PR #9 `5d363f5`, PR #7 `2e10e6a` và PR #1 `2406b12`; sau phép thử chính còn ghép PR #11 `d4b2697` để kiểm tra liên nhóm. Mỗi phép merge cục bộ đều sạch. Không merge hoặc push `main` hay branch thành viên. Checkout NV4 gốc vẫn giữ các bản sao checkpoint/mã mô hình chưa stage từ lần kiểm thử trước; commit NV4 chỉ chứa tài liệu và ảnh chụp.

## Phạm vi commit NV3 mới

So với `e8b0e6f`, commit `2e10e6a` **không sửa** `src/models/resnet18.py`, `configs/resnet18.yaml`, `src/data/transforms.py`, `scripts/infer_resnet18.py` hoặc checkpoint. `src/resnet18_workflow.py:126` mở rộng `calculate_metrics` với số mẫu, macro precision/recall và chỉ số từng lớp; không đổi `load_checkpoint`/`predict_image`. `scripts/eval_resnet18.py:27` từ chối ghi đè kết quả cũ trước khi tải checkpoint; dòng 46–55 thêm SHA-256, split, nhãn, preprocessing và metadata checkpoint vào JSON; CSV có cột `split`. Test NV3 được bổ sung cho các hợp đồng này. Không phát hiện thay đổi kiến trúc, thứ tự logits hoặc pipeline suy luận.

## Checkpoint và suy luận thật

- `checkpoints/resnet18_best.pt` lấy từ gói Drive được NV3 công bố ở PR #7, SHA-256 tính lại `61d44ba8ac02ee57c44cbd3835aaac2c657c97924735fea743a60df5f5fdf9e2`. Checkpoint CNN của PR #9 có SHA-256 `d33c95bbe15e82fa83f1fd954136ebfff6e1cbba1a86fe40af473a377390de1f`.
- Sidecar ResNet18: bốn nhãn `Healthy`, `Early Blight`, `Late Blight`, `Bacterial Spot`; fine-tuning epoch 9, Validation macro-F1 `0.9808585369397294`. YAML/sidecar/`class_mapping.json` khớp. Transform Val/Test: resize cạnh ngắn 256, center crop 224, ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- `torch.load(weights_only=True)` + `load_state_dict(strict=True)` với `build_model(4)` của HEAD NV3 đạt trên CPU; `eval()`, input `[1,3,224,224]`, logits `[1,4]`.

| Ảnh dataset thật | Nhãn thật | ResNet18 | Xác suất theo thứ tự bốn lớp | Đúng |
|---|---|---|---|---|
| `000146ff...259.1.JPG` | Healthy | Healthy | 0,999775 / 0,000006 / 0,000209 / 0,000009 | Có |
| `0012b9d2...8389.JPG` | Early Blight | Early Blight | 0,000026 / 0,999882 / 0,000080 / 0,000012 | Có |
| `0003faa8...4946.JPG` | Late Blight | Late Blight | 0,000000 / 0,006698 / 0,993301 / 0,000000 | Có |
| `00416648...3110.JPG` | Bacterial Spot | Bacterial Spot | 0,000783 / 0,255758 / 0,072370 / 0,671089 | Có |

`python -m demo.check_consistency <ảnh Bacterial Spot thật> --model both`: **CNN PASS** và **ResNet18 PASS**, top-1 Bacterial Spot, `max |delta|=0` cho cả hai. Bản mới của `scripts.eval_resnet18` đã chạy lại toàn bộ 1.356 ảnh Test: Accuracy `0.9867256637168141`, macro-F1 `0.983977947579615`, JSON có SHA-256/epoch/Validation metadata, CSV có 1.356 hàng và cột `split=test`. Chạy lại vào cùng thư mục bị từ chối đúng bằng `FileExistsError` ở `scripts/eval_resnet18.py:27`, bảo vệ các artifact đã có.

## Streamlit trên worktree tích hợp

Ứng dụng chạy tại `127.0.0.1:8511` bằng Python trong `.venv` dự án. Trình duyệt upload hai ảnh thật Bacterial Spot và Healthy, xem trước và thử lần lượt ba lựa chọn:

| Chế độ | Bacterial Spot | Healthy | Bốn xác suất |
|---|---|---|---|
| CNN | Bacterial Spot 96,95% | Healthy 99,96% | Đủ trên từng ảnh |
| ResNet18 | Bacterial Spot 67,11% | Healthy 99,98% | Đủ trên từng ảnh |
| So sánh | Hai mô hình hiển thị song song, cùng nhãn Bacterial Spot | Hai mô hình hiển thị song song, cùng nhãn Healthy | Đủ cho mỗi mô hình |

Tải thêm `bad_image.jpg` (nội dung không phải JPEG) vào batch: ảnh lỗi có thông báo riêng; hai ảnh thật vẫn được dự đoán ở chế độ so sánh. Lịch sử trong phiên hiển thị kết quả và cho xem lại. CSV tải trực tiếp từ trình duyệt có BOM UTF-8 `EF-BB-BF`, 10 cột theo hợp đồng và 8 hàng đầu tiên đúng tên ảnh/mô hình/kết quả. Dashboard sau kiểm thử hiển thị **2 ảnh riêng biệt, 12 lượt suy luận, 1 lượt lỗi** và ghi rõ đây là thống kê nhãn dự đoán. Ảnh chụp thật: [so sánh](../../demo/screenshots/nv3_head_2e10_compare_20261101.jpg), [Dashboard](../../demo/screenshots/nv3_head_2e10_dashboard_20261101.jpg), [ảnh lỗi](../../demo/screenshots/nv3_head_2e10_bad_image_20261101.jpg).

## Bộ test và giới hạn

- Worktree main + PR #9/#7/#1: `python -m pytest -q` → **76 passed, 1 skipped, 19 subtests passed**, 0 failed.
- Thêm PR #11 HEAD để thử liên nhóm: **86 passed, 1 skipped, 19 subtests passed**, 0 failed. Một test NV2 skip có chủ ý vì mã dữ liệu NV1 đã hiện diện. Có 16 cảnh báo deprecation Matplotlib/NumPy, không có test fail.
- `python -m compileall -q src demo scripts tests` và `git diff --check` đạt. Không phát hiện lỗi NV4 hoặc NV3 cần sửa. Không thêm regression test vì không có lỗi hồi quy thực tế.
- Kết quả là trên worktree ghép PR, **chưa phải kiểm thử từ `main` sau merge**. PR #7/#9/#1 còn mở; Gate 4 vẫn cần review, tích hợp chính thức, kiểm thử lại `main` và video demo cuối nếu nhóm yêu cầu.
