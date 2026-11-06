"""Streamlit presentation for the group's verified tomato leaf classifiers."""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("Thiếu Streamlit. Cài đặt bằng pip install -r requirements.txt") from exc

from demo.session import CLASS_LABELS_VI, MAX_UPLOADS, MODEL_LABELS, append_history, dashboard, export_csv, predict_uploads, selected_models
from src.inference import InferenceError, MAX_IMAGE_BYTES, MODELS, ROOT, class_names, load_config, predict_image, read_image
from src.view_geometry import draw_field_of_view, geometry_warnings, model_field_of_view


def preprocessing_for(models):
    """Preprocessing per model; a model whose config cannot be read is simply skipped here."""
    result = {}
    for model in models:
        try:
            result[model] = load_config(model)["preprocessing"]
        except (InferenceError, KeyError, OSError):
            continue
    return result


def explain_image(*args, **kwargs):
    # Import inside the optional step so an unavailable visualizer cannot break classification.
    from src.explainability import explain_image as explain
    return explain(*args, **kwargs)


st.set_page_config(page_title="Dự đoán bệnh lá cà chua", page_icon="🍅", layout="wide")
st.markdown("""<style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: rgba(72, 150, 95, 0.13); border: 1px solid rgba(72, 150, 95, 0.35);
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
    show_attention = st.checkbox(
        "Hiển thị vùng mô hình chú ý (Grad-CAM – thử nghiệm)", value=False,
        help="Đây là vùng mô hình chú ý, KHÔNG phải vùng bệnh được xác định bằng nhãn vị trí.",
    )
    uploads = st.file_uploader("Ảnh JPG, JPEG hoặc PNG; tối đa 10 MiB mỗi ảnh", type=("jpg", "jpeg", "png"), accept_multiple_files=True)
    selection = (choice, device, show_attention, tuple((upload.name, hashlib.sha256(upload.getvalue()).hexdigest()) for upload in uploads))
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
                        st.image(read_image(upload.getvalue()), caption=upload.name, width="stretch", alt=f"Ảnh tải lên: {upload.name}")
                    except InferenceError as exc:
                        st.error(f"{upload.name}: {exc}")
    if st.button("Dự đoán bệnh", type="primary", disabled=len(uploads) > MAX_UPLOADS, width="stretch"):
        st.session_state.nv4_results = []
        if not uploads:
            st.error("Vui lòng tải ít nhất một ảnh JPG hoặc PNG trước khi dự đoán.")
        else:
            progress = st.progress(0, text="Đang xử lý ảnh...")
            items, records = [], []
            for index, upload in enumerate(uploads, 1):
                batch_items, batch_records = predict_uploads(
                    [upload], selected_models(choice), predict_image, device=device,
                    explain_fn=explain_image if show_attention else None,
                )
                items.extend(batch_items)
                records.extend(batch_records)
                progress.progress(index / len(uploads), text=f"Đã xử lý {index}/{len(uploads)} ảnh")
            progress.empty()
            st.session_state.nv4_results = items
            st.session_state.nv4_history = append_history(st.session_state.nv4_history, records)
    if st.session_state.nv4_results:
        st.subheader("Kết quả lượt này")
        for index, item in enumerate(st.session_state.nv4_results):
            with st.container(border=True):
                st.subheader(item["name"])
                if "image" in item["errors"]:
                    st.error(item["errors"]["image"])
                    continue
                original = read_image(uploads[index].getvalue())
                preprocessing = preprocessing_for(selected_models(choice))
                warnings = geometry_warnings(original.width, original.height, preprocessing)
                if warnings:
                    st.warning("Ảnh dài/rộng so với khung mô hình. " + " ".join(warnings)
                               + " Nên cắt ảnh sát lá, gần vuông, rồi tải lại để kết quả đáng tin hơn.")
                resnet_view = (model_field_of_view(original.width, original.height, "resnet18", preprocessing["resnet18"])
                               if "resnet18" in preprocessing else None)
                if resnet_view and resnet_view["coverage"] < 0.999:
                    st.image(draw_field_of_view(original, resnet_view["box"]), width="stretch",
                             caption="Ảnh gốc – khung xanh/vùng sáng là phần ResNet18 phân tích; CNN dùng cả ảnh (bị ép vuông)",
                             alt=f"Ảnh gốc có đánh dấu vùng ResNet18 phân tích: {item['name']}")
                else:
                    st.image(original, caption="Ảnh gốc", width="stretch", alt=f"Ảnh gốc: {item['name']}")
                # Each model column holds its own Grad-CAM and result, so stacked mobile layouts stay paired.
                columns = st.columns(len(selected_models(choice)))
                for column, model in zip(columns, selected_models(choice)):
                    with column:
                        st.markdown(f"#### {MODEL_LABELS[model]}")
                        if show_attention:
                            attention = item.get("attention", {}).get(model)
                            if attention and attention["png"] is not None:
                                st.image(attention["png"], width="stretch",
                                         caption=f"Grad-CAM {MODEL_LABELS[model]} – ảnh 224×224 mô hình nhìn thấy",
                                         alt=f"Vùng chú ý xấp xỉ của {MODEL_LABELS[model]}")
                            elif attention:
                                st.info(attention["message"])
                            else:
                                st.caption("Không có Grad-CAM do bước dự đoán chưa thành công.")
                        if model in item["errors"]:
                            message = item["errors"][model]
                            if message.startswith("Thiếu checkpoint"):
                                message = f"{MODEL_LABELS[model]} chưa sẵn sàng: thiếu checkpoint được nhóm xác nhận."
                            st.warning(message)
                        elif model in item["results"]:
                            result = item["results"][model]
                            attention = item.get("attention", {}).get(model)
                            label = result["top_label"]
                            st.success(f"Dự đoán: {CLASS_LABELS_VI.get(label, label)} ({label})")
                            st.table({"Lớp": [f"{CLASS_LABELS_VI.get(name, name)} ({name})" for name in result["probabilities"]],
                                      "Xác suất": [f"{value:.2%}" for value in result["probabilities"].values()]})
                            st.bar_chart({"Lớp": list(result["probabilities"]), "Xác suất": list(result["probabilities"].values())}, x="Lớp", y="Xác suất", horizontal=True)
                            if attention:
                                st.caption(attention["message"])
                                st.caption("Vùng đánh dấu thể hiện nơi mô hình chú ý, không phải vị trí bệnh đã được xác thực.")
                                if attention["png"] is not None:
                                    st.download_button(
                                        f"Tải ảnh kết quả {MODEL_LABELS[model]}", attention["png"],
                                        file_name=f"{Path(item['name']).stem}_{model}_gradcam.png", mime="image/png",
                                        key=f"gradcam_download_{index}_{model}", on_click="ignore", width="stretch",
                                    )
                            st.caption(f"Forward pass: {result['forward_ms']:.2f} ms")
    elif uploads:
        st.caption("Nhấn Dự đoán bệnh để xem kết quả cho các ảnh đã chọn.")

with tab_history:
    records = st.session_state.nv4_history
    st.caption("Chỉ lưu tên ảnh và kết quả trong phiên trình duyệt; không lưu nội dung ảnh vào lịch sử.")
    if records:
        names = class_names()
        st.dataframe([{"Thời gian": row["predicted_at"], "Ảnh": row["image_name"], "Mô hình": row["model"],
                       "Trạng thái": row.get("status", "Thành công"), "Nhãn": row["top_label"],
                       "Top-1": f'{row["top_probability"]:.2%}' if row.get("top_probability") is not None else "—"}
                      for row in records], width="stretch", hide_index=True)
        selected_index = st.selectbox("Xem lại kết quả", range(len(records)), format_func=lambda index: f'{records[index]["image_name"]} · {records[index]["model"]} · {records[index]["predicted_at"]}')
        selected = records[selected_index]
        if selected.get("status", "Thành công") == "Lỗi":
            st.error(selected["error"])
        else:
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
    model_filter = st.selectbox("Lọc theo mô hình", ("Tất cả", "CNN", "ResNet18"))
    summary = dashboard(st.session_state.nv4_history, names, None if model_filter == "Tất cả" else model_filter)
    metrics = st.columns(4)
    metrics[0].metric("Ảnh đã dự đoán", summary["images"])
    metrics[1].metric("Lượt suy luận", summary["predictions"])
    metrics[2].metric("Lượt lỗi", summary["errors"])
    metrics[3].metric("Forward trung bình", f'{summary["mean_forward_ms"]:.2f} ms' if summary["mean_forward_ms"] is not None else "—")
    if not st.session_state.nv4_history:
        st.info("Chưa có dữ liệu trong phiên. Hãy dự đoán ảnh để xem thống kê.")
    left, right = st.columns(2)
    with left:
        st.subheader("Lượt theo mô hình")
        st.bar_chart({"Mô hình": list(summary["by_model"]), "Lượt": list(summary["by_model"].values())}, x="Mô hình", y="Lượt")
    with right:
        st.subheader("Nhãn được dự đoán")
        st.bar_chart({"Nhãn": list(summary["by_label"]), "Lượt": list(summary["by_label"].values())}, x="Nhãn", y="Lượt")
    st.caption("Đây là thống kê nhãn mô hình dự đoán trong phiên, không phải tỷ lệ bệnh thực tế trong vườn.")
