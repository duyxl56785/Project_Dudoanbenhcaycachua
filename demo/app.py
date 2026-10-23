"""Streamlit demo for group supplied tomato leaf classifiers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Thiếu Streamlit. Cài đặt bằng pip install -r requirements.txt") from exc

from src.inference import InferenceError, MAX_IMAGE_BYTES, class_names, predict_bytes, read_image

st.set_page_config(page_title="Phân loại lá cà chua", page_icon="🍅")
st.title("Phân loại tình trạng lá cà chua")
st.write("Tải ảnh JPEG/PNG để xem dự đoán từ CNN, ResNet18 hoặc so sánh hai mô hình.")
st.info("Chỉ phân loại bốn lớp đã học. Ảnh bệnh hoặc cây khác vẫn có thể bị gán vào một lớp. PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát. Kết quả không phải chẩn đoán chắc chắn và không thay thế chuyên gia nông nghiệp. Xác suất softmax chưa được hiệu chỉnh.")
choice = st.radio("Mô hình", ("CNN", "ResNet18", "So sánh cả hai"), horizontal=True)
upload = st.file_uploader("Ảnh JPG/PNG (tối đa 10 MiB)", type=["jpg", "jpeg", "png"])
if upload is not None:
    data = upload.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        st.error("Ảnh vượt giới hạn 10 MiB.")
    else:
        try:
            image = read_image(data)
            names = class_names()
            st.image(image, caption="Ảnh đã tải", use_container_width=True)
            selected = ("cnn", "resnet18") if choice == "So sánh cả hai" else (choice.lower(),)
            for model_name in selected:
                st.subheader("CNN" if model_name == "cnn" else "ResNet18")
                try:
                    result = predict_bytes(data, model_name)
                    st.write(f"Dự đoán: **{result['top_label']}**")
                    st.table({"Lớp": names, "Xác suất softmax": [f"{result['probabilities'][name]:.2%}" for name in names]})
                    st.caption(f"Forward pass: {result['forward_ms']:.2f} ms; không gồm tải mô hình, đọc ảnh hoặc tiền xử lý.")
                except InferenceError as exc:
                    st.error(str(exc))
        except InferenceError as exc:
            st.error(str(exc))
