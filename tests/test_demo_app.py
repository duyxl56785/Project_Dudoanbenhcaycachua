"""Exercise the result UI, comparison, downloads and graceful visualization failure."""
import io
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest
from src.inference import ROOT


def test_results_comparison_downloads_history_and_visualization_failure():
    data = io.BytesIO()
    Image.new("RGB", (300, 180), "green").save(data, format="PNG")
    upload = SimpleNamespace(name="leaf.png", size=len(data.getvalue()), getvalue=data.getvalue)
    result = {"top_index": 0, "top_label": "Healthy", "probabilities": {
        "Healthy": 0.7, "Early Blight": 0.1, "Late Blight": 0.1, "Bacterial Spot": 0.1}, "forward_ms": 1.0}
    attention = {"png": data.getvalue(), "has_box": False, "message": "Healthy không vẽ khung."}
    with patch("streamlit.file_uploader", return_value=[upload]), \
         patch("src.inference.predict_image", return_value=result), \
         patch("src.explainability.explain_image", return_value=attention) as explain:
        app = AppTest.from_file(ROOT / "demo/app.py", default_timeout=15).run()
        assert not app.exception
        app.radio[0].set_value("So sánh cả hai")
        app.checkbox[0].check()
        app.button[0].click().run()
        assert not app.exception
        assert len(app.session_state["nv4_history"]) == 2
        assert sum("Lá khỏe mạnh (Healthy)" in element.value for element in app.success) == 2
        assert len(app.image) == 4  # Preview + original + both overlays.
        assert len(app.get("download_button")) == 3  # Two PNGs and CSV.
        assert all("attention" not in record for record in app.session_state["nv4_history"])
        explain.side_effect = ImportError("visualizer unavailable")
        app.button[0].click().run()
        assert not app.exception
        assert len(app.session_state["nv4_history"]) == 4
        assert len(app.session_state["nv4_results"][0]["results"]) == 2
        assert any("visualizer unavailable" in element.value for element in app.info)
        app.checkbox[0].uncheck()
        app.button[0].click().run()
        assert not app.exception
        assert "attention" not in app.session_state["nv4_results"][0]
        assert len(app.session_state["nv4_history"]) == 6
