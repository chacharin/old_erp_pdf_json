"""
app.py — Flask middleware: รับ PDF จาก user → แปลงเป็นภาพ → ส่ง LLM (OpenRouter)
→ ตรวจ JSON → คืนผลลัพธ์ + บันทึก log การทดลอง

วิธีรัน:
    pip install -r requirements.txt
    python app.py
แล้วเปิด http://127.0.0.1:5000
"""

import csv
import io
import json
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

import json_validator
import llm_client
import logger as exp_logger
import pdf_module

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.json.sort_keys = False  # คงลำดับ key ตามคอลัมน์มาตรฐาน ไม่ให้ jsonify เรียงใหม่  # 20 MB

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path(__file__).parent / "outputs_json"  # JSON ผลลัพธ์ของแต่ละการรัน

COLUMN_ORDER = json_validator.REQUIRED_KEYS  # ลำดับคอลัมน์มาตรฐาน 11 ช่อง


@app.get("/")
def index():
    return render_template("index.html", models=llm_client.MODELS)


@app.post("/api/extract")
def extract():
    t0 = time.perf_counter()
    run_id = exp_logger.new_run_id()

    # --- รับและตรวจไฟล์ ---
    file = request.files.get("pdf")
    model = (request.form.get("model") or "").strip()
    if not file or not file.filename:
        return jsonify({"error": "กรุณาเลือกไฟล์ PDF"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "รองรับเฉพาะไฟล์ .pdf"}), 400
    if not model:
        return jsonify({"error": "กรุณาเลือกหรือพิมพ์ชื่อโมเดล"}), 400

    filename = secure_filename(file.filename) or "upload.pdf"
    pdf_path = UPLOAD_DIR / f"{run_id}_{filename}"
    file.save(pdf_path)

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": run_id,
        "pdf_file": filename,
        "model": model,
    }

    try:
        # --- 1) PDF → ภาพ ---
        conv = pdf_module.pdf_to_images(str(pdf_path))
        record["pages"] = conv["pages"]
        record["time_convert_s"] = conv["time_convert_s"]

        # --- 2) ส่ง LLM ---
        llm = llm_client.extract_from_images(conv["data_urls"], model)
        record["time_llm_s"] = llm["time_llm_s"]
        record["prompt_tokens"] = llm["prompt_tokens"]
        record["completion_tokens"] = llm["completion_tokens"]
        record["total_tokens"] = llm["total_tokens"]

        # --- 3) ตรวจผลลัพธ์ ---
        report = json_validator.validate(llm["content"])

        # จัดเรียง key ทุกแถวให้ตรงลำดับคอลัมน์มาตรฐาน (key เกินต่อท้าย)
        if report["data"]:
            report["data"] = [
                {**{k: row.get(k, "-") for k in COLUMN_ORDER},
                 **{k: v for k, v in row.items() if k not in COLUMN_ORDER}}
                if isinstance(row, dict) else row
                for row in report["data"]
            ]

        # เซฟ JSON output เป็นไฟล์ แล้วบันทึกชื่อไฟล์ลง log
        output_file = ""
        if report["data"] is not None:
            OUTPUT_DIR.mkdir(exist_ok=True)
            output_name = f"{Path(filename).stem}_{run_id}.json"
            (OUTPUT_DIR / output_name).write_text(
                json.dumps(report["data"], ensure_ascii=False, indent=2),
                encoding="utf-8")
            output_file = f"outputs_json/{output_name}"
        record["output_file"] = output_file

        record["valid_json"] = report["valid_json"]
        record["rows_extracted"] = report["rows_extracted"]
        record["arithmetic_pass"] = (
            "" if report["arithmetic_pass"] is None else report["arithmetic_pass"])
        record["time_total_s"] = round(time.perf_counter() - t0, 3)
        record["error"] = report["parse_error"] or ""

        exp_logger.log_experiment(record)
        exp_logger.save_raw_response(run_id, {
            "record": record,
            "model_requested": model,
            "model_actual": llm["model"],
            "raw_content": llm["content"],
            "validation": {k: v for k, v in report.items() if k != "data"},
        })

        return jsonify({
            "run_id": run_id,
            "model": llm["model"],
            "pages": conv["pages"],
            "output_file": output_file,
            "timing": {
                "convert_s": conv["time_convert_s"],
                "llm_s": llm["time_llm_s"],
                "total_s": record["time_total_s"],
            },
            "usage": {
                "prompt_tokens": llm["prompt_tokens"],
                "completion_tokens": llm["completion_tokens"],
                "total_tokens": llm["total_tokens"],
            },
            "validation": {
                "valid_json": report["valid_json"],
                "stripped_fences": report["stripped_fences"],
                "key_errors": report["key_errors"],
                "arithmetic_pass": report["arithmetic_pass"],
                "arithmetic_errors": report["arithmetic_errors"],
                "parse_error": report["parse_error"],
            },
            "rows_extracted": report["rows_extracted"],
            "data": report["data"],
            "raw_content": llm["content"],
        })

    except (pdf_module.PdfConversionError, llm_client.LlmError) as e:
        record["time_total_s"] = round(time.perf_counter() - t0, 3)
        record["error"] = str(e)
        exp_logger.log_experiment(record)
        return jsonify({"error": str(e), "run_id": run_id}), 502
    finally:
        try:
            pdf_path.unlink(missing_ok=True)  # ไม่เก็บ PDF ไว้หลังประมวลผล
        except OSError:
            pass  # ลบไม่ได้ก็ข้าม (เช่นไฟล์ถูกล็อก) ไม่ให้กระทบผลลัพธ์


@app.get("/api/logs")
def logs():
    return jsonify(exp_logger.read_logs())


@app.post("/api/export-csv")
def export_csv():
    """รับ JSON array (ผลที่สกัดได้) → คืนไฟล์ CSV (UTF-8 BOM เปิดใน Excel ได้ทันที)"""
    payload = request.get_json(silent=True) or {}
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return jsonify({"error": "ไม่มีข้อมูลสำหรับ export"}), 400

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMN_ORDER, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        if isinstance(row, dict):
            writer.writerow({k: row.get(k, "-") for k in COLUMN_ORDER})

    fname = secure_filename(str(payload.get("filename") or "output")) or "output"
    csv_bytes = buf.getvalue().encode("utf-8-sig")  # ใส่ BOM ให้ Excel แสดงภาษาไทยถูก
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename={fname}.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
