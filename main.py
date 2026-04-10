import base64
import datetime as dt
import io
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import face_recognition
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "user_data.pkl"
TOLERANCE = 0.52  #人脸识别最低阈值
ENROLL_DUPLICATE_TOLERANCE = 0.45  #防重复录入阈值
ALLOWED_GENDERS = {"男", "女", "其他"}

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def empty_db() -> Dict[str, Any]:
    return {"users": [], "meta": {"next_id": 1, "version": 1}}


def load_db() -> Dict[str, Any]:
    if not DB_PATH.exists():
        return empty_db()
    with DB_PATH.open("rb") as f:
        db = pickle.load(f)
    if "users" not in db or "meta" not in db:
        return empty_db()
    if "next_id" not in db["meta"]:
        db["meta"]["next_id"] = max([u["id"] for u in db["users"]] + [0]) + 1
    return db


def save_db(db: Dict[str, Any]) -> None:
    with DB_PATH.open("wb") as f:
        pickle.dump(db, f)


def image_bytes_to_array(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}") from exc
    return np.array(image)


def extract_single_encoding(image_np: np.ndarray) -> np.ndarray:
    locations = face_recognition.face_locations(image_np)
    if len(locations) == 0:
        raise HTTPException(status_code=400, detail="图片中未检测到人脸")
    encodings = face_recognition.face_encodings(image_np, known_face_locations=locations)
    if len(encodings) == 0:
        raise HTTPException(status_code=400, detail="人脸特征提取失败")
    return encodings[0]


def get_user_by_id(db: Dict[str, Any], user_id: int) -> Optional[Dict[str, Any]]:
    for user in db["users"]:
        if user["id"] == user_id:
            return user
    return None


def to_public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": user["id"],
        "name": user["name"],
        "gender": user["gender"],
        "class_name": user["class_name"],
        "enroll_time": user["enroll_time"],
        "image_url": user["image_url"],
    }


def validate_profile(name: str, gender: str, class_name: str) -> None:
    if not name.strip():
        raise HTTPException(status_code=400, detail="姓名不能为空")
    if gender not in ALLOWED_GENDERS:
        raise HTTPException(status_code=400, detail="性别不合法")
    if not class_name.strip():
        raise HTTPException(status_code=400, detail="班级不能为空")


def match_encoding(test_encoding: np.ndarray, db: Dict[str, Any]) -> Dict[str, Any]:
    if not db["users"]:
        return {"name": "未知", "distance": None, "user_id": None}

    known_encodings = [np.array(user["face_encoding"]) for user in db["users"]]
    distances = face_recognition.face_distance(known_encodings, test_encoding)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    if best_dist < TOLERANCE:
        matched_user = db["users"][best_idx]
        return {
            "name": matched_user["name"],
            "distance": best_dist,
            "user_id": matched_user["id"],
        }
    return {"name": "未知", "distance": best_dist, "user_id": None}


def find_duplicate_user(test_encoding: np.ndarray, db: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not db["users"]:
        return None
    known_encodings = [np.array(user["face_encoding"]) for user in db["users"]]
    distances = face_recognition.face_distance(known_encodings, test_encoding)
    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])
    if best_dist < ENROLL_DUPLICATE_TOLERANCE:
        user = db["users"][best_idx]
        return {"user_id": user["id"], "name": user["name"], "distance": best_dist}
    return None


class UserUpdateBody(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None
    class_name: Optional[str] = None


app = FastAPI(title="Face Identify System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/data/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


INDEX_HTML_PATH = BASE_DIR / "index.html"


@app.get("/")
def root() -> FileResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="未找到 index.html")
    return FileResponse(str(INDEX_HTML_PATH), media_type="text/html")


@app.get("/index.html")
def index_html() -> FileResponse:
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="未找到 index.html")
    return FileResponse(str(INDEX_HTML_PATH), media_type="text/html")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/enroll")
async def enroll_user(
    name: str = Form(...),
    gender: str = Form(...),
    class_name: str = Form(...),
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    validate_profile(name, gender, class_name)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="未上传图片")

    image_np = image_bytes_to_array(image_bytes)
    encoding = extract_single_encoding(image_np)

    db = load_db()
    duplicate = find_duplicate_user(encoding, db)
    if duplicate is not None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"该人脸已存在（匹配用户：{duplicate['name']}，"
                f"相似度距离：{duplicate['distance']:.4f}），不能重复录入"
            ),
        )

    user_id = db["meta"]["next_id"]
    image_suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    image_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}{image_suffix}"
    image_path = IMAGES_DIR / image_filename
    with image_path.open("wb") as f:
        f.write(image_bytes)

    user = {
        "id": user_id,
        "name": name.strip(),
        "gender": gender,
        "class_name": class_name.strip(),
        "enroll_time": dt.datetime.now().isoformat(timespec="seconds"),
        "image_path": str(image_path),
        "image_url": f"/data/images/{image_filename}",
        "face_encoding": encoding.tolist(),
    }
    db["users"].append(user)
    db["meta"]["next_id"] = user_id + 1
    save_db(db)
    return {"message": "录入成功", "user": to_public_user(user)}


@app.get("/api/users")
def list_users() -> Dict[str, Any]:
    db = load_db()
    users = [to_public_user(user) for user in db["users"]]
    return {"total": len(users), "users": users}


@app.get("/api/users/{user_id}")
def get_user(user_id: int) -> Dict[str, Any]:
    db = load_db()
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"user": to_public_user(user)}


@app.put("/api/users/{user_id}")
async def update_user(
    user_id: int,
    name: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    class_name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    db = load_db()
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    new_name = name.strip() if name is not None else user["name"]
    new_gender = gender if gender is not None else user["gender"]
    new_class = class_name.strip() if class_name is not None else user["class_name"]
    validate_profile(new_name, new_gender, new_class)

    user["name"] = new_name
    user["gender"] = new_gender
    user["class_name"] = new_class

    if file is not None:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="上传图片为空")
        image_np = image_bytes_to_array(image_bytes)
        encoding = extract_single_encoding(image_np)

        old_image_path = Path(user["image_path"])
        if old_image_path.exists():
            old_image_path.unlink()

        image_suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
        image_filename = f"user_{user_id}_{uuid.uuid4().hex[:8]}{image_suffix}"
        image_path = IMAGES_DIR / image_filename
        with image_path.open("wb") as f:
            f.write(image_bytes)

        user["image_path"] = str(image_path)
        user["image_url"] = f"/data/images/{image_filename}"
        user["face_encoding"] = encoding.tolist()

    save_db(db)
    return {"message": "更新成功", "user": to_public_user(user)}


@app.delete("/api/users/all")
def delete_all_users() -> Dict[str, Any]:
    db = load_db()
    deleted = 0
    for user in db["users"]:
        image_path = Path(user["image_path"])
        if image_path.exists():
            image_path.unlink()
        deleted += 1
    db["users"] = []
    db["meta"]["next_id"] = 1
    save_db(db)
    return {"message": "已全部删除", "deleted": deleted}


@app.delete("/api/users/{user_id}")
def delete_user(user_id: int) -> Dict[str, str]:
    db = load_db()
    user = get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")

    image_path = Path(user["image_path"])
    if image_path.exists():
        image_path.unlink()
    db["users"] = [u for u in db["users"] if u["id"] != user_id]
    save_db(db)
    return {"message": "删除成功"}


@app.post("/api/recognize/image")
async def recognize_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="未上传图片")

    image_np = image_bytes_to_array(image_bytes)
    locations = face_recognition.face_locations(image_np)
    encodings = face_recognition.face_encodings(image_np, known_face_locations=locations)

    db = load_db()
    faces = []
    for encoding, location in zip(encodings, locations):
        top, right, bottom, left = location
        match = match_encoding(encoding, db)
        faces.append(
            {
                "bbox": {"top": top, "right": right, "bottom": bottom, "left": left},
                "name": match["name"],
                "distance": match["distance"],
                "user_id": match["user_id"],
            }
        )
    return {"count": len(faces), "faces": faces}


def decode_ws_image(binary_data: bytes) -> np.ndarray:
    if not binary_data:
        raise ValueError("空帧")
    return image_bytes_to_array(binary_data)


@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            image_bytes: Optional[bytes] = None
            message = await websocket.receive()
            msg_type = message.get("type")
            if msg_type == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                image_bytes = message["bytes"]
            elif message.get("text"):
                text_data = message["text"]
                if "," in text_data and text_data.startswith("data:image"):
                    text_data = text_data.split(",", 1)[1]
                try:
                    image_bytes = base64.b64decode(text_data)
                except Exception:
                    await websocket.send_json({"error": "文本帧不是有效的 base64 图片数据"})
                    continue
            else:
                continue

            try:
                image_np = decode_ws_image(image_bytes)
                small_frame = image_np
                max_width = 640
                if image_np.shape[1] > max_width:
                    ratio = max_width / image_np.shape[1]
                    resized = Image.fromarray(image_np).resize(
                        (int(image_np.shape[1] * ratio), int(image_np.shape[0] * ratio))
                    )
                    small_frame = np.array(resized)
                scale_x = image_np.shape[1] / small_frame.shape[1]
                scale_y = image_np.shape[0] / small_frame.shape[0]

                locations = face_recognition.face_locations(small_frame)
                encodings = face_recognition.face_encodings(small_frame, known_face_locations=locations)
                db = load_db()
                faces = []
                for encoding, location in zip(encodings, locations):
                    top, right, bottom, left = location
                    match = match_encoding(encoding, db)
                    faces.append(
                        {
                            "bbox": {
                                "top": int(top * scale_y),
                                "right": int(right * scale_x),
                                "bottom": int(bottom * scale_y),
                                "left": int(left * scale_x),
                            },
                            "name": match["name"],
                            "distance": match["distance"],
                            "user_id": match["user_id"],
                        }
                    )
                await websocket.send_json({"faces": faces, "count": len(faces)})
            except Exception as exc:
                try:
                    await websocket.send_json({"error": f"实时识别处理失败: {exc}"})
                except Exception:
                    break
                continue
    except WebSocketDisconnect:
        return
    except Exception:
        # 出现未知异常时直接结束本次连接，避免影响其他连接。
        return


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
