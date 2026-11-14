# Z-Image Studio

Web app chạy local (React + FastAPI) dùng **[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)** để tạo ảnh từ thư viện tham chiếu: **khuôn mặt, trang phục, tư thế, bối cảnh**. Chọn nhiều ảnh ở mỗi nhóm → app tạo **mọi tổ hợp × số ảnh mỗi tổ hợp** trong một lần bấm (tối đa 64 ảnh), chạy tuần tự trên GPU.

## Cách hoạt động (quan trọng)

Z-Image-Turbo là model **text-to-image**, không nhận ảnh tham chiếu trực tiếp. App xử lý như sau:

| Nhóm | Cách đưa vào model |
|---|---|
| **Tư thế** | Ảnh → skeleton OpenPose (nút *Trích tư thế*) → **ControlNet** [`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union`](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0). Bám tư thế khá chính xác. |
| **Khuôn mặt / Trang phục / Bối cảnh** | Mỗi ảnh có một **mô tả văn bản** (tự viết, hoặc bấm *Tự mô tả* để Qwen2.5-VL-3B viết) và mô tả được ghép vào prompt. |

⇒ Trang phục và bối cảnh ra khá giống. **Khuôn mặt chỉ giống ở mức đặc điểm** (tuổi, tóc, dáng mặt…), *không giữ đúng danh tính 100%*. Muốn giữ đúng mặt một người cụ thể cần train LoRA cho Z-Image hoặc dùng model chỉnh sửa ảnh theo tham chiếu.

## Yêu cầu

- GPU NVIDIA, khuyến nghị ≥ 16 GB VRAM (12 GB dùng `ZIMAGE_OFFLOAD=model`). Lần đầu tải model ~20 GB từ Hugging Face.
- Python 3.10+, Node.js 18+.

## Cài đặt

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

```bash
cd frontend
npm install
```

## Chạy

```bash
./start.sh          # Windows: start.bat
```

Mở http://localhost:5173. Hoặc build frontend một lần (`cd frontend && npm run build`) rồi chỉ cần chạy backend và mở http://localhost:8000.

## Sử dụng

1. **Thư viện** → chọn nhóm → kéo thả ảnh vào.
   - Mặt / trang phục / bối cảnh: bấm **Tự mô tả** (hoặc tự gõ mô tả tiếng Anh) → sửa → **Lưu**.
   - Tư thế: bấm **Trích tư thế**, hoặc tick “ảnh đã là skeleton” trước khi tải lên.
2. **Tạo ảnh** → bấm chọn một hoặc nhiều ảnh mỗi nhóm (bỏ trống = không dùng), viết prompt chính, chọn kích thước, số ảnh/tổ hợp → **Tạo**.
   - Prompt có thể đặt chỗ: `A photo of {face}, wearing {outfit}, in {background}`. Không dùng chỗ đặt thì mô tả tự nối vào sau prompt.
   - Mỗi ảnh dùng seed = seed gốc + thứ tự, nên nhập lại seed để tái tạo kết quả.
3. **Kết quả**: xem, copy prompt, tải xuống, xoá. Ảnh lưu ở `backend/data/outputs/`.

## Cấu hình (biến môi trường của backend)

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `ZIMAGE_MODEL` | `Tongyi-MAI/Z-Image-Turbo` | Model hoặc đường dẫn local |
| `ZIMAGE_OFFLOAD` | `none` | `model` / `sequential` để tiết kiệm VRAM |
| `ZIMAGE_PRELOAD` | `1` | Tải model ngay khi khởi động |
| `ZIMAGE_CONTROLNET_REPO` / `ZIMAGE_CONTROLNET_FILE` | Union 2.0 repo / `...-2.1.safetensors` | File ControlNet |
| `CAPTION_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Model mô tả ảnh (tải khi bấm lần đầu) |
| `CAPTION_KEEP_LOADED` | `0` | `1` = giữ model mô tả trong VRAM |
| `MAX_JOBS_PER_REQUEST` | `64` | Giới hạn ảnh mỗi lần tạo |
| `DATA_DIR` | `backend/data` | Nơi lưu thư viện & kết quả |

Thông số khuyến nghị cho Turbo: 9 steps, guidance 0 (app đặt sẵn).
