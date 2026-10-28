"""Streamlit presentation for the group's verified tomato leaf classifiers."""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Thiếu Streamlit. Cài đặt bằng pip install -r requirements.txt") from exc

from demo.session import MAX_UPLOADS, MODEL_LABELS, dashboard, export_csv, predict_uploads, selected_models
from src.inference import InferenceError, MAX_IMAGE_BYTES, MODELS, ROOT, class_names, predict_image, read_image

st.set_page_config(page_title="Dự đoán bệnh lá cà chua", page_icon="🍅", layout="wide")
st.markdown("""<style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f3f8f5; border: 1px solid #dce9df;
      border-radius: 12px; padding: 1rem;}
    @media (max-width: 640px) {.block-container {padding: 1rem .75rem;}}
</style>""", unsafe_allow_html=True)

if "nv4_history" not in st.session_state:
    st.session_state.nv4_history = []
if "nv4_results" not in st.session_state:
    st.session_state.nv4_results = []

with st.sidebar:
    st.header("Thiết lập")
    choice = st.radio("Mô hình", ("CNN", "ResNet18", "So sánh cả hai"))
    device_choice = st.selectbox("Thiết bị suy luận", ("Tự động", "CPU", "CUDA"))
    device = {"Tự động": "auto", "CPU": "cpu", "CUDA": "cuda"}[device_choice]
    st.divider()
    st.subheader("Trạng thái mô hình")
    for model, filename in MODELS.items():
        if (ROOT / "checkpoints" / filename).is_file():
            st.success(f"{MODEL_LABELS[model]}: có checkpoint")
        else:
            st.warning(f"{MODEL_LABELS[model]}: chưa sẵn sàng ({filename})")

st.title("🍅 Dự đoán bệnh lá cà chua")
st.write("Tải tối đa tám ảnh để xem kết quả CNN, ResNet18 hoặc so sánh hai mô hình trên cùng ảnh.")
st.info("Chỉ phân loại bốn lớp đã học. PlantVillage chủ yếu là ảnh trong điều kiện kiểm soát. Softmax chưa được hiệu chỉnh và kết quả không thay thế chuyên gia nông nghiệp.")
tab_predict, tab_history, tab_dashboard = st.tabs(("Dự đoán", "Lịch sử phiên", "Thống kê phiên"))

with tab_predict:
    st.subheader("Tải ảnh")
    uploads = st.file_uploader("Ảnh JPG, JPEG hoặc PNG; tối đa 10 MiB mỗi ảnh", type=("jpg", "jpeg", "png"), accept_multiple_files=True)
    selection = (choice, device, tuple((upload.name, hashlib.sha256(upload.getvalue()).hexdigest()) for upload in uploads))
    if st.session_state.get("nv4_selection") != selection:
        st.session_state.nv4_results = []
        st.session_state.nv4_selection = selection
    if len(uploads) > MAX_UPLOADS:
        st.error(f"Chỉ tải tối đa {MAX_UPLOADS} ảnh mỗi lượt. Hãy bỏ bớt ảnh trước khi dự đoán.")
    elif uploads:
        st.caption(f"Đã chọn {len(uploads)}/{MAX_UPLOADS} ảnh")
        for upload in uploads:
            st.write(f"• {upload.name}")
        with st.expander("Xem trước ảnh", expanded=True):
            columns = st.columns(2)
            for index, upload in enumerate(uploads):
                with columns[index % 2]:
                    try:
                        if upload.size > MAX_IMAGE_BYTES:
                            raise InferenceError("Ảnh vượt giới hạn 10 MiB.")
                        st.image(read_image(upload.getvalue()), caption=upload.name, width="stretch")
                    except InferenceError as exc:
                        st.error(f"{upload.name}: {exc}")
    if st.button("Dự đoán bệnh", type="primary", disabled=len(uploads) > MAX_UPLOADS):
        st.session_state.nv4_results = []
        if not uploads:
            st.error("Vui lòng tải ít nhất một ảnh JPG hoặc PNG trước khi dự đoán.")
        else:
            with st.spinner("Đang xử lý ảnh..."):
                items, records = predict_uploads(uploads, selected_models(choice), predict_image, device=device)
            st.session_state.nv4_results = items
            st.session_state.nv4_history.extend(records)
    if st.session_state.nv4_results:
        st.subheader("Kết quả lượt này")
        for item in st.session_state.nv4_results:
            with st.container(border=True):
                st.markdown(f"#### {item['name']}")
                if "image" in item["errors"]:
                    st.error(item["errors"]["image"])
                    continue
                columns = st.columns(len(selected_models(choice)))
                for column, model in zip(columns, selected_models(choice)):
                    with column:
                        st.markdown(f"**{MODEL_LABELS[model]}**")
                        if model in item["errors"]:
                            message = item["errors"][model]
                            if message.startswith("Thiếu checkpoint"):
                                message = f"{MODEL_LABELS[model]} chưa sẵn sàng: thiếu checkpoint được nhóm xác nhận."
                            st.warning(message)
                        elif model in item["results"]:
                            result = item["results"][model]
                            st.success(f"Dự đoán: {result['top_label']}")
                            st.table({"Lớp": list(result["probabilities"]), "Xác suất": [f"{value:.2%}" for value in result["probabilities"].values()]})
                            st.bar_chart({"Lớp": list(result["probabilities"]), "Xác suất": list(result["probabilities"].values())}, x="Lớp", y="Xác suất", horizontal=True)
                            st.caption(f"Forward pass: {result['forward_ms']:.2f} ms")
    elif uploads:
        st.caption("Nhấn Dự đoán bệnh để xem kết quả cho các ảnh đã chọn.")

with tab_history:
    records = st.session_state.nv4_history
    st.caption("Chỉ lưu tên ảnh và kết quả trong phiên trình duyệt; không lưu nội dung ảnh vào lịch sử.")
    if records:
        names = class_names()
        st.dataframe([{"Thời gian": row["predicted_at"], "Ảnh": row["image_name"], "Mô hình": row["model"], "Nhãn": row["top_label"], "Top-1": f'{row["top_probability"]:.2%}'} for row in records], width="stretch", hide_index=True)
        selected_index = st.selectbox("Xem lại kết quả", range(len(records)), format_func=lambda index: f'{records[index]["image_name"]} · {records[index]["model"]} · {records[index]["predicted_at"]}')
        selected = records[selected_index]
        st.write(f"**{selected['top_label']}** · {selected['top_probability']:.2%}")
        st.table({"Lớp": names, "Xác suất": [f'{selected["probabilities"][name]:.2%}' for name in names]})
        st.download_button("Tải kết quả CSV", export_csv(records, names), file_name="du_doan_la_ca_chua.csv", mime="text/csv")
        if st.button("Xóa lịch sử"):
            st.session_state.nv4_history = []
            st.session_state.nv4_results = []
            st.rerun()
    else:
        st.write("Chưa có dự đoán trong phiên này.")

with tab_dashboard:
    names = class_names()
    summary = dashboard(st.session_state.nv4_history, names)
    metrics = st.columns(3)
    metrics[0].metric("Ảnh đã dự đoán", summary["images"])
    metrics[1].metric("Lượt suy luận", summary["predictions"])
    metrics[2].metric("Forward trung bình", f'{summary["mean_forward_ms"]:.2f} ms' if summary["mean_forward_ms"] is not None else "—")
    left, right = st.columns(2)
    with left:
        st.subheader("Lượt theo mô hình")
        st.bar_chart({"Mô hình": list(summary["by_model"]), "Lượt": list(summary["by_model"].values())}, x="Mô hình", y="Lượt")
    with right:
        st.subheader("Nhãn được dự đoán")
        st.bar_chart({"Nhãn": list(summary["by_label"]), "Lượt": list(summary["by_label"].values())}, x="Nhãn", y="Lượt")
    st.caption("Đây là thống kê nhãn mô hình dự đoán trong phiên, không phải tỷ lệ bệnh thực tế trong vườn.")
