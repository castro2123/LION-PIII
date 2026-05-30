import io
import pandas as pd
import streamlit as st
from PIL import Image

from yolo_comparison import (
    run_yolo_lion_comparison,
    build_consensus_ground_truth,
    export_results_to_pdf,
)

MODEL_NAMES   = ["YOLOv8s", "YOLOv11s", "YOLOv12s", "LION"]
_SS_RESULTS   = "yolo_cmp_results"
_SS_ANNOTATED = "yolo_cmp_annotated"
_SS_IMG_KEY   = "yolo_cmp_img_key"


def _init_ss() -> None:
    for key in (_SS_RESULTS, _SS_ANNOTATED, _SS_IMG_KEY):
        if key not in st.session_state:
            st.session_state[key] = None


def _clear_results() -> None:
    st.session_state[_SS_RESULTS]   = None
    st.session_state[_SS_ANNOTATED] = None


def _grid_cols(n: int):
    if n <= 3:
        return list(st.columns(n))
    half = (n + 1) // 2
    return list(st.columns(half)) + list(st.columns(n - half))


# ─────────────────────────────────────────────────────────────────
# Render
# ─────────────────────────────────────────────────────────────────
def render_yolo_comparison() -> None:
    _init_ss()

    st.header("YOLO Comparison · v8 · v11 · v12 + LION")
    st.caption(
        "YOLO detects objects · LION enriches with captions · "
        "LION standalone can also detect directly."
    )

    # ── Imagem ─────────────────────────────────────────────────────
    img: Image.Image | None = st.session_state.get("img")
    if img is None:
        st.warning("Upload an image first.")
        return

    current_key = hash(img.tobytes())
    if st.session_state[_SS_IMG_KEY] != current_key:
        _clear_results()
        st.session_state[_SS_IMG_KEY] = current_key

    # ── Parâmetros ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        conf_threshold = st.slider("Confidence", 0.10, 0.95, 0.25, 0.05)
    with col2:
        iou_threshold  = st.slider("IoU (NMS)",  0.10, 0.95, 0.45, 0.05)
    with col3:
        show_conf = st.toggle("Show confidence", value=True)
    with col4:
        use_lion_caption = st.toggle("🦁 LION captions", value=True,
            help="Use LION to describe each YOLO detection.")

    # ── Modelos ────────────────────────────────────────────────────
    st.markdown("### Models")
    chk_cols     = st.columns(len(MODEL_NAMES))
    active_flags = {}
    for col, name in zip(chk_cols, MODEL_NAMES):
        with col:
            active_flags[name] = st.checkbox(name, value=True)

    selected_names = [n for n, v in active_flags.items() if v]
    if not selected_names:
        st.warning("Select at least one model.")
        return

    # Consenso automático entre modelos — sem GT externo necessário
    ground_truth = None
    include_lion_standalone = "LION" in selected_names

    # ── Botão ──────────────────────────────────────────────────────
    if st.button("▶ Compare", type="primary"):
        from model_registry import get_models
        all_models = get_models()

        registry_keys = {"YOLOv8s": "yolo_v8", "YOLOv11s": "yolo_v11", "YOLOv12s": "yolo_v12"}
        yolo_models = {}
        for display_name, reg_key in registry_keys.items():
            if display_name in selected_names:
                if reg_key not in all_models:
                    st.error(f"Missing model in registry: {display_name}")
                    return
                yolo_models[display_name] = all_models[reg_key]

        lion      = all_models.get("lion")
        processor = all_models.get("processor")
        device    = all_models.get("device", "cpu")

        if use_lion_caption and lion is None:
            st.warning("LION model not available — captions disabled.")

        with st.spinner("Running comparison…"):
            output = run_yolo_lion_comparison(
                yolo_models=yolo_models, lion=lion, processor=processor,
                device=device, img=img, conf=conf_threshold, iou=iou_threshold,
                show_conf=show_conf, use_lion_caption=use_lion_caption,
                include_lion_standalone=include_lion_standalone,
                ground_truth=ground_truth,
            )

        st.session_state[_SS_RESULTS]   = output
        st.session_state[_SS_ANNOTATED] = output["annotated"]

    # ── Resultados ─────────────────────────────────────────────────
    output = st.session_state[_SS_RESULTS]
    if not output:
        return

    results           = output["results"]
    annotated         = output["annotated"]
    comparison_table  = output["comparison_table"]
    diff_table        = output["diff_table"]
    metrics_table     = output["metrics_table"]
    metrics_mode      = output["metrics_mode"]
    per_class_metrics = output["per_class_metrics"]
    confidence_table  = output["confidence_table"]

    # ── Cards ──────────────────────────────────────────────────────
    st.divider()
    cols = _grid_cols(len(results))
    for col, (name, (dets, ms)) in zip(cols, results.items()):
        with col:
            st.metric(
                f"{'🦁' if name == 'LION' else '🔵'} {name}",
                f"{len(dets)} objects", f"{ms:.0f} ms",
            )

    # ── Imagens anotadas ───────────────────────────────────────────
    st.subheader("🖼 Detections")
    cols = _grid_cols(len(results))
    for col, (name, (dets, ms)) in zip(cols, results.items()):
        with col:
            st.image(annotated[name],
                caption=f"{name} · {len(dets)} objects · {ms:.0f} ms",
                use_container_width=True)

    # ── Métricas principais ────────────────────────────────────────
    st.divider()

    mode_info = {
        "ground_truth": ("📈 Metrics  —  vs Ground Truth",
            "Precision, Recall, F1 and mAP@50 computed against your annotations."),
        "consensus":    ("📈 Metrics  —  Cross-Model Consensus",
            "Precision, Recall, F1 and mAP@50 computed automatically: detections agreed by ≥50% of models form the reference. The more models agree, the more reliable the metrics."),
        "confidence":   ("📈 Confidence Statistics",
            "ℹ️ Only one model selected and no ground truth — showing confidence distribution."),
    }
    title, note = mode_info.get(metrics_mode, ("📈 Metrics", ""))
    st.subheader(title)
    if note:
        st.caption(note)

    st.dataframe(pd.DataFrame(metrics_table), use_container_width=True, hide_index=True)

    # ── Métricas por classe ────────────────────────────────────────
    if per_class_metrics:
        with st.expander("📚 Per-class metrics (Precision / Recall / F1 / AP@50)"):
            st.dataframe(
                pd.DataFrame(per_class_metrics),
                use_container_width=True, hide_index=True,
            )

    # ── Confiança por classe ───────────────────────────────────────
    with st.expander("🎯 Confidence by class"):
        st.dataframe(
            pd.DataFrame(confidence_table),
            use_container_width=True, hide_index=True,
        )

    # ── Contagem por classe ────────────────────────────────────────
    st.subheader("📊 Count by class")
    st.dataframe(pd.DataFrame(comparison_table), use_container_width=True, hide_index=True)

    # ── Divergências ───────────────────────────────────────────────
    st.subheader("⚡ Divergences")
    if diff_table:
        st.dataframe(pd.DataFrame(diff_table), use_container_width=True, hide_index=True)
    else:
        st.success("All models detected the same classes.")

    # ── Detalhes por modelo ────────────────────────────────────────
    st.subheader("🔍 Detection details")
    cols = _grid_cols(len(results))
    for col, (name, (dets, _)) in zip(cols, results.items()):
        with col:
            with st.expander(f"{name} ({len(dets)} detections)"):
                if not dets:
                    st.write("_No detections._")
                else:
                    rows = [{
                        "Label":      d["label"],
                        "Caption":    d.get("caption", ""),
                        "Confidence": f"{d.get('confidence', 1.0):.1%}",
                        "BBox":       d["bbox"],
                    } for d in sorted(dets, key=lambda x: -x.get("confidence", 1.0))]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Exportar ───────────────────────────────────────────────────
    st.divider()
    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        csv_buf = io.StringIO()
        pd.DataFrame(comparison_table).to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Export CSV",
            data=csv_buf.getvalue(),
            file_name="yolo_comparison.csv",
            mime="text/csv",
        )

    with exp_col2:
        # Resolve o GT para o PDF da mesma forma que o pipeline
        gt_for_pdf = ground_truth
        if gt_for_pdf is None and len(results) >= 2:
            gt_for_pdf = build_consensus_ground_truth(results, iou_threshold)

        pdf_bytes = export_results_to_pdf(
            results_per_model=results,
            annotated_images=annotated,
            ground_truth=gt_for_pdf,
            iou_threshold=iou_threshold,
        )
        st.download_button(
            "📄 Export PDF Report",
            data=pdf_bytes,
            file_name="yolo_comparison_report.pdf",
            mime="application/pdf",
        )