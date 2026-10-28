"""Streamlit demo for group supplied tomato leaf classifiers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Thiếu Streamlit. Cài đặt bằng pip install -r requirements.txt") from exc

from src.inference import InferenceError, MAX_IMAGE_BYTES, ROOT, MODELS, class_names, predict_image, read_image

st.set_page_config(page_title="Phân loại lá cà chua", page_icon="🍅")
st.title("Phân loại tình trạng lá cà chua")
st.write("Đồ án phân loại ảnh lá cà chua bằng CNN tự xây dựng và ResNet18 Transfer Learning. Tải một ảnh để xem dự đoán hoặc so sánh hai mô hình trên cùng ảnh.")
st.info("Chỉ phân loại bốn lớp đã học. Ảnh bệnh hoặc cây khác vẫn có thể bị gán vào một lớp. PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát. Kết quả không phải chẩn đoán chắc chắn và không thay thế chuyên gia nông nghiệp. Xác suất softmax chưa được hiệu chỉnh.")
for model_name, checkpoint_name in MODELS.items():
    if not (ROOT / "checkpoints" / checkpoint_name).is_file():
        st.warning(f"{'CNN' if model_name == 'cnn' else 'ResNet18'}: chưa có checkpoint được nhóm xác nhận ({checkpoint_name}).")
choice = st.radio("Mô hình", ("CNN", "ResNet18", "So sánh cả hai"), horizontal=True)
device_choice = st.selectbox("Thiết bị suy luận", ("Tự động", "CPU", "CUDA"))
device = {"Tự động": "auto", "CPU": "cpu", "CUDA": "cuda"}[device_choice]
upload = st.file_uploader("Ảnh JPG/PNG (tối đa 10 MiB)", type=["jpg", "jpeg", "png"])
if upload is not None:
    data = upload.getvalue()
    if len(data) > MAX_IMAGE_BYTES:
        st.error("Ảnh vượt giới hạn 10 MiB.")
    else:
        try:
            image = read_image(data)
            names = class_names()
            st.image(image, caption="Ảnh đã tải", width="stretch")
            selected = ("cnn", "resnet18") if choice == "So sánh cả hai" else (choice.lower(),)
            if st.button("Dự đoán", type="primary"):
                for model_name in selected:
                    st.subheader("CNN" if model_name == "cnn" else "ResNet18")
                    try:
                        result = predict_image(image, model_name, device=device)
                        st.write(f"Dự đoán: **{result['top_label']}**")
                        st.table({"Lớp": names, "Xác suất softmax": [f"{result['probabilities'][name]:.2%}" for name in names]})
                        st.bar_chart({"Lớp": list(names), "Xác suất": [result["probabilities"][name] for name in names]}, x="Lớp", y="Xác suất", horizontal=True)
                        st.caption(f"Forward pass: {result['forward_ms']:.2f} ms trên {device_choice}; không gồm tải mô hình, đọc ảnh hoặc tiền xử lý.")
                    except InferenceError as exc:
                        st.error(str(exc))
        except InferenceError as exc:
            st.error(str(exc))
