import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from gel_core import analyze_gel, overlay_results, load_config


st.set_page_config(page_title="Gel Dilution Recommender", layout="wide")

st.title("Gel Dilution Recommender")
st.caption("Row별로 1000bp 기준선을 직접 그려서 반영할 수 있습니다.")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def image_to_png_bytes(img) -> bytes:
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("PNG 인코딩에 실패했습니다.")
    return buf.tobytes()


def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def draw_saved_anchor_lines(base_bgr: np.ndarray, anchor_store: dict, row_h: int) -> np.ndarray:
    img = base_bgr.copy()

    for row_num, item in anchor_store.items():
        if not item.get("enabled"):
            continue
        if item.get("y") is None:
            continue

        y_abs = int((row_num - 1) * row_h + float(item["y"]))
        cv2.line(img, (20, y_abs), (img.shape[1] - 20, y_abs), (0, 255, 255), 2)
        cv2.putText(
            img,
            f"Row {row_num} - 1000bp",
            (30, max(20, y_abs - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img




def draw_selected_row_preview(base_bgr: np.ndarray, selected_row: int, row_h: int, preview_y: float | None = None) -> np.ndarray:
    img = base_bgr.copy()
    y0 = int((selected_row - 1) * row_h)
    y1 = int(selected_row * row_h - 1)
    cv2.rectangle(img, (6, max(0, y0)), (img.shape[1] - 6, min(img.shape[0] - 1, y1)), (255, 180, 0), 2)
    if preview_y is not None:
        y_abs = int(y0 + float(preview_y))
        cv2.line(img, (20, y_abs), (img.shape[1] - 20, y_abs), (255, 0, 255), 2)
        cv2.putText(
            img,
            f"Preview Row {selected_row}: y={preview_y:.1f}",
            (30, max(24, y_abs - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return img

def build_manual_anchor_dict(anchor_store: dict, default_lane: int) -> dict:
    manual = {}
    for row_num, info in anchor_store.items():
        if info.get("enabled") and info.get("y") is not None:
            manual[int(row_num)] = {
                "lane": int(info.get("lane", default_lane)),
                "y": float(info["y"]),
            }
    return manual

def extract_latest_line_y(canvas_result, selected_row: int, row_h: int, display_scale: float = 1.0):
    if canvas_result is None:
        return None

    json_data = canvas_result.json_data
    if not json_data:
        return None

    objects = json_data.get("objects", [])
    if not objects:
        return None

    line_objects = [obj for obj in objects if obj.get("type") == "line"]
    if not line_objects:
        return None

    latest = line_objects[-1]

    top = float(latest.get("top", 0))
    y1 = float(latest.get("y1", 0))
    y2 = float(latest.get("y2", 0))

    abs_y1_display = top + y1
    abs_y2_display = top + y2
    y_abs_display = (abs_y1_display + abs_y2_display) / 2.0

    y_abs_original = y_abs_display / max(display_scale, 1e-6)

    row_y0 = (selected_row - 1) * row_h
    row_y1 = selected_row * row_h

    if not (row_y0 <= y_abs_original < row_y1):
        return None

    y_in_row = y_abs_original - row_y0
    y_in_row = max(0.0, min(float(row_h - 1), y_in_row))
    return y_in_row

st.sidebar.header("설정")
target_mode = st.sidebar.selectbox("Target mode", ["AUTO", "ITS", "16S"], index=0)
config_path = st.sidebar.text_input("Config path", value="config.yaml")

uploaded_file = st.file_uploader(
    "Gel image 업로드",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("분석할 gel 이미지를 업로드하세요.")
    st.stop()

temp_path = None

try:
    temp_path = save_uploaded_file(uploaded_file)

    cfg = load_config(config_path)
    n_rows = int(cfg["rows_to_split"])
    lane_count = int(cfg["lane_count_per_row"])
    ladder_lanes = cfg.get("ladder_lanes", [9, 18])
    default_lane = int(ladder_lanes[0]) if ladder_lanes else 1

    if "anchor_store" not in st.session_state:
        st.session_state.anchor_store = {
            r: {"enabled": False, "lane": default_lane, "y": None}
            for r in range(1, n_rows + 1)
        }

    existing_rows = set(st.session_state.anchor_store.keys())
    needed_rows = set(range(1, n_rows + 1))
    if existing_rows != needed_rows:
        st.session_state.anchor_store = {
            r: st.session_state.anchor_store.get(
                r,
                {"enabled": False, "lane": default_lane, "y": None}
            )
            for r in range(1, n_rows + 1)
        }

    raw_bytes = uploaded_file.getvalue()
    np_bytes = np.frombuffer(raw_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError("업로드한 이미지를 읽을 수 없습니다.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_rgb.shape[:2]
    row_h = img_h // n_rows

    canvas_target_width = 1100
    display_scale = min(1.0, float(canvas_target_width) / float(img_w))
    canvas_w = max(400, int(img_w * display_scale))
    canvas_h = max(200, int(img_h * display_scale))
    st.sidebar.markdown("---")
    st.sidebar.subheader("수동 1000bp anchor")

    selected_row = st.sidebar.selectbox("선택 Row", list(range(1, n_rows + 1)), index=0)

    save_line_button = st.sidebar.button("현재 그린 선을 선택 Row anchor로 저장", use_container_width=True)
    clear_selected = st.sidebar.button("선택 Row anchor 삭제", use_container_width=True)
    clear_all = st.sidebar.button("전체 anchor 삭제", use_container_width=True)
    run_button = st.sidebar.button("분석 실행", type="primary", use_container_width=True)

    if clear_selected:
        st.session_state.anchor_store[selected_row] = {
            "enabled": False,
            "lane": default_lane,
            "y": None
        }

    if clear_all:
        st.session_state.anchor_store = {
            r: {"enabled": False, "lane": default_lane, "y": None}
            for r in range(1, n_rows + 1)
        }

    display_bgr = draw_saved_anchor_lines(img_bgr, st.session_state.anchor_store, row_h)
    display_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
    display_pil = Image.fromarray(display_rgb).resize(
        (canvas_w, canvas_h),
        Image.Resampling.BILINEAR
    )

    col1, col2 = st.columns([1.35, 0.65])

    with col1:
        st.subheader("원본 이미지 + 저장된 1000bp 선")
        st.write(f"현재 선택 Row: {selected_row}")
        st.write("이미지 위에 직선을 하나 그린 뒤, 왼쪽의 저장 버튼을 누르세요.")
        st.write("선은 반드시 선택한 Row 안에 그려야 합니다.")

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 0, 0.0)",
            stroke_width=3,
            stroke_color="#FFFF00",
            background_image=display_pil,
            update_streamlit=True,
            height=canvas_h,
            width=canvas_w,
            drawing_mode="line",
            point_display_radius=0,
            display_toolbar=False,
            key=f"canvas_row_{selected_row}_{uploaded_file.name}",
        )

        latest_y = extract_latest_line_y(
            canvas_result,
            selected_row,
            row_h,
            display_scale=display_scale
        )

        if latest_y is not None:
            st.info(f"현재 그린 선의 선택 Row 내부 y값: {latest_y:.1f}px")
            preview_bgr = draw_selected_row_preview(display_bgr, selected_row, row_h, latest_y)
            st.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), caption="선택 Row 및 현재 anchor 미리보기", width=min(1200, canvas_w))
        else:
            st.info("아직 선택 Row 안에 저장 가능한 선이 없습니다.")
            preview_bgr = draw_selected_row_preview(display_bgr, selected_row, row_h, None)
            st.image(cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB), caption="선택 Row 강조 미리보기", width=min(1200, canvas_w))

        if save_line_button:
            if latest_y is None:
                st.warning("선택한 Row 안에 직선을 먼저 그려주세요.")
            else:
                st.session_state.anchor_store[selected_row] = {
                    "enabled": True,
                    "lane": default_lane,
                    "y": float(latest_y),
                }
                st.success(
                    f"Row {selected_row}의 1000bp가 저장되었습니다. "
                    f"(y_in_row={latest_y:.1f})"
                )
                st.rerun()

    with col2:
        st.subheader("입력 정보")
        st.write(f"파일명: {uploaded_file.name}")
        st.write(f"Target mode: {target_mode}")
        st.write(f"Config: {config_path}")
        st.write(f"Rows: {n_rows}")
        st.write(f"Lanes per row: {lane_count}")
        st.write(f"Ladder lanes: {ladder_lanes}")

        st.markdown("### 저장된 수동 anchor")
        for r in range(1, n_rows + 1):
            item = st.session_state.anchor_store[r]
            if item["enabled"] and item["y"] is not None:
                st.write(f"Row {r}: y={float(item['y']):.1f}px")
            else:
                st.write(f"Row {r}: 없음")

        st.markdown("### 사용 순서")
        st.write("1. 왼쪽에서 Row를 선택")
        st.write("2. 이미지 위에 직선을 그림")
        st.write("3. 저장 버튼 클릭")
        st.write("4. 다른 Row도 반복")
        st.write("5. 분석 실행 클릭")

    manual_anchors_1000 = build_manual_anchor_dict(
        st.session_state.anchor_store,
        default_lane=default_lane,
    )

    if run_button:
        with st.spinner("분석 중입니다..."):
            final_analysis = analyze_gel(
                image_path=temp_path,
                target_mode=target_mode,
                config_path=config_path,
                manual_anchors_1000=manual_anchors_1000,
            )

            df = final_analysis["df"]
            color_img = final_analysis["color_img"]
            overlay_img = overlay_results(color_img, df)
            overlay_img = draw_saved_anchor_lines(overlay_img, st.session_state.anchor_store, row_h)

            csv_bytes = dataframe_to_csv_bytes(df)
            png_bytes = image_to_png_bytes(overlay_img)

        st.success("분석이 완료되었습니다.")

        result_preview_width = min(1400, max(canvas_w, 1200))
        st.image(
            overlay_img[:, :, ::-1],
            caption="분석 결과 오버레이",
            width=result_preview_width
        )
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="결과 PNG 다운로드",
                data=png_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_{target_mode}_result.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                label="결과 CSV 다운로드",
                data=csv_bytes,
                file_name=f"{Path(uploaded_file.name).stem}_{target_mode}_result.csv",
                mime="text/csv",
                use_container_width=True,
            )

finally:
    if temp_path and os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except Exception:
            pass