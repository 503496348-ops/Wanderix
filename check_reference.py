#!/usr/bin/env python3
"""
参考图监控预检脚本
在调用 API 前对参考图进行全面诊断：
  1. 文件存在性 + 大小
  2. OpenCV 人脸检测（多策略）
  3. 裁切区域预览（可视化诊断）
  4. Base64 编码还原度
  5. 最终建议（是否适合作为参考图）
  6. 生成优化后的参考图路径（若需要）

用法:
  python3 check_reference.py /path/to/photo.jpg
  python3 check_reference.py /path/to/photo.jpg --fix   # 自动修复并保存到 /tmp
"""
import sys
import os
import json
import base64
import tempfile
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ─── 配置 ───────────────────────────────────────────────────────────────
MIN_FACE_PIXELS = 30          # 人脸最小边长(px)
MIN_FACE_RATIO = 0.002       # 人脸最小占图比例（防止误检）
TARGET_FACE_RATIO = 0.03     # 目标人脸占图比例（3%）
AUTO_CROP_SIZE = (1024, 1536) # 裁切后目标分辨率（2:3）


# ─── 诊断报告 ────────────────────────────────────────────────────────
class ReferenceDiagnostic:
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = None
        self.h = self.w = 0
        self.gray = None
        self.faces_default = []
        self.faces_strict = []
        self.best_face = None
        self.crop_region = None
        self.errors = []
        self.warnings = []
        self.ok = True

    def load(self) -> bool:
        """加载图片"""
        try:
            self.img = cv2.imread(self.img_path)
            if self.img is None:
                self.errors.append(f"无法读取图片: {self.img_path}")
                return False
            self.h, self.w = self.img.shape[:2]
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            return True
        except Exception as e:
            self.errors.append(f"读取异常: {e}")
            return False

    def check_file(self) -> dict:
        """Step 1: 文件检查"""
        p = Path(self.img_path)
        size_kb = p.stat().st_size // 1024
        size_mb = size_kb / 1024

        info = {
            "path": str(p),
            "size_kb": size_kb,
            "size_mb": round(size_mb, 2),
            "dimensions": f"{self.w}×{self.h}",
            "total_pixels": self.w * self.h,
            "large_file": size_kb > 8192,
        }
        if size_kb > 8192:
            self.warnings.append(f"文件大小 {size_mb:.1f} MB > 8MB，将触发压缩")
        return info

    def detect_faces(self) -> dict:
        """Step 2: 人脸检测（多策略）"""
        casc_path = cv2.data.haarcascades
        cascades = {
            "default":  casc_path + "haarcascade_frontalface_default.xml",
            "alt2":     casc_path + "haarcascade_frontalface_alt2.xml",
            "profile":  casc_path + "haarcascade_profileface.xml",
        }

        results = {}
        for name, xml in cascades.items():
            if not Path(xml).exists():
                results[name] = {"count": 0, "faces": []}
                continue
            clf = cv2.CascadeClassifier(xml)
            faces = clf.detectMultiScale(self.gray, 1.1, 4)
            results[name] = {"count": len(faces), "faces": faces.tolist() if len(faces) > 0 else []}

        self.faces_default = results["default"]["faces"]
        return results

    def pick_best_face(self) -> dict:
        """Step 3: 从候选中选择最佳人脸（多策略组合）"""
        if len(self.faces_default) == 0:
            self.warnings.append("Haar 未检测到人脸，可能需要手动标注或换图")
            return {"strategy": "none", "face": None}

        total_area = self.w * self.h
        candidates = []

        for x, y, fw, fh in self.faces_default:
            area = fw * fh
            y_ratio = y / self.h if self.h > 0 else 0
            aspect = fw / fh if fh > 0 else 0
            area_ratio = area / total_area

            # score_by_area
            score_by_area = area

            # score_by_pos
            y_center_penalty = abs(y_ratio - 0.5)  # 离中心越远惩罚越大
            score_by_pos = area * (1 - y_center_penalty * 1.2)

            # score_shape
            shape_ok = 0.8 <= aspect <= 1.25
            score_shape = area * (2 if shape_ok else 0.5)

            # combined score
            combined_score = score_by_area * (0.3 if shape_ok else 0.1) + score_by_pos * 0.4 + score_shape * 0.3

            candidates.append({
                "x": x, "y": y, "fw": fw, "fh": fh,
                "area": area,
                "y_ratio": round(y_ratio, 3),
                "aspect": round(aspect, 2),
                "area_ratio": round(area_ratio * 100, 3),  # 百分比
                "score_by_area": score_by_area,
                "score_by_pos": round(score_by_pos, 0),
                "combined_score": round(combined_score, 0),
            })

        candidates.sort(key=lambda c: c["combined_score"], reverse=True)
        best = candidates[0]
        self.best_face = best

        return {
            "strategy": "综合评分（位置+面积+形状）",
            "top_candidates": candidates[:5],
            "best": best
        }

    def compute_crop(self) -> dict:
        """Step 4: 计算裁切区域"""
        if self.best_face is None:
            return {"cropped": False, "reason": "无人脸"}

        x, y, fw, fh = self.best_face["x"], self.best_face["y"], self.best_face["fw"], self.best_face["fh"]
        total_area = self.w * self.h
        face_ratio = (fw * fh) / total_area

        # 判断是否需要裁切
        needs_crop = face_ratio < TARGET_FACE_RATIO

        # 计算裁切区域
        crop_top = max(0, y - int(fh * 0.6))
        crop_bottom = min(self.h, y + fh + int(fh * 2.2))
        ratio = 2 / 3
        crop_h = crop_bottom - crop_top
        crop_w = int(crop_h * ratio)

        cx = x + fw // 2
        left = max(0, cx - crop_w // 2)
        right = min(self.w, cx + crop_w // 2)

        if left == 0 or right == self.w:
            left, right = 0, self.w
            crop_top = max(0, crop_bottom - int(self.w / ratio))

        crop_img = self.img[crop_top:crop_bottom, left:right]
        nh, nw = crop_img.shape[:2]
        new_face_ratio = (fw * fh) / (nw * nh)

        self.crop_region = {
            "original": {"w": self.w, "h": self.h, "face_ratio": round(face_ratio * 100, 3)},
            "crop": {"x1": left, "y1": crop_top, "x2": right, "y2": crop_bottom, "w": nw, "h": nh},
            "face_in_crop": round(new_face_ratio * 100, 2),
            "needs_crop": needs_crop,
            "face": {"x": x, "y": y, "fw": fw, "fh": fh}
        }

        return self.crop_region

    def check_base64_fidelity(self, img_bytes: bytes) -> dict:
        """Step 5: Base64 编码还原度"""
        encoded = base64.b64encode(img_bytes).decode("ascii")
        decoded = base64.b64decode(encoded)

        # 比较像素
        arr_orig = np.frombuffer(img_bytes, dtype=np.uint8)
        arr_dec = np.frombuffer(decoded, dtype=np.uint8)

        return {
            "original_size": len(img_bytes),
            "base64_length": len(encoded),
            "decoded_size": len(decoded),
            "decoding_ok": arr_orig.tobytes() == arr_dec.tobytes(),
            "lossless": arr_orig.tobytes() == arr_dec.tobytes(),
        }

    def generate_report(self) -> str:
        """生成完整诊断报告"""
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"📋 参考图诊断报告")
        lines.append(f"{'='*60}")

        # 文件信息
        finfo = self.check_file()
        lines.append(f"\n📁 文件: {finfo['path']}")
        lines.append(f"   尺寸: {finfo['dimensions']} | 像素: {finfo['total_pixels']:,}")
        lines.append(f"   大小: {finfo['size_mb']} MB ({finfo['size_kb']} KB)")
        if finfo["large_file"]:
            lines.append(f"   ⚠️  超过 8MB，需要压缩")

        # 人脸检测
        detect = self.detect_faces()
        lines.append(f"\n🔍 人脸检测:")
        lines.append(f"   默认级联: 检测到 {detect['default']['count']} 个候选")
        lines.append(f"   严格级联: 检测到 {detect['alt2']['count']} 个候选")
        lines.append(f"   侧脸级联: 检测到 {detect['profile']['count']} 个候选")

        # 最佳人脸
        picker = self.pick_best_face()
        best = picker.get("best")
        if best:
            lines.append(f"\n🏆 最佳人脸（{picker['strategy']}）:")
            lines.append(f"   位置: x={best['x']} y={best['y']} w={best['fw']} h={best['fh']}")
            lines.append(f"   y占比: {best['y_ratio']*100:.1f}% | 宽高比: {best['aspect']} | 面积占比: {best['area_ratio']}%")
            lines.append(f"   综合评分: {best['combined_score']}")
            lines.append(f"   TOP5 候选:")
            for i, c in enumerate(picker["top_candidates"][:5]):
                lines.append(f"     #{i+1}: y_ratio={c['y_ratio']} w={c['fw']} h={c['fh']} 占比{c['area_ratio']}% score={c['combined_score']}")
        else:
            lines.append(f"\n⚠️  未选择到最佳人脸")

        # 裁切
        crop = self.compute_crop()
        if crop.get("needs_crop") or crop.get("face"):
            lines.append(f"\n📐 裁切分析:")
            orig = crop["original"]
            c = crop["crop"]
            lines.append(f"   原图: {orig['w']}×{orig['h']}，人脸占图 {orig['face_ratio']}%")
            lines.append(f"   裁区: ({c['x1']},{c['y1']}) → ({c['x2']},{c['y2']}) = {c['w']}×{c['h']}")
            lines.append(f"   人脸在裁切后占比: {crop['face_in_crop']}%")
            if crop["needs_crop"]:
                lines.append(f"   ✅ 建议裁切（人脸占比 {orig['face_ratio']}% < {TARGET_FACE_RATIO*100}% 阈值）")
            else:
                lines.append(f"   ℹ️  无需裁切（人脸占比已达标）")

        # 错误和警告
        if self.errors:
            lines.append(f"\n❌ 错误:")
            for e in self.errors:
                lines.append(f"   {e}")
            self.ok = False

        if self.warnings:
            lines.append(f"\n⚠️  警告:")
            for w in self.warnings:
                lines.append(f"   {w}")

        # 总结
        lines.append(f"\n{'='*60}")
        if self.errors:
            lines.append(f"❌ 诊断未通过，请修复上述错误后重试")
        elif detect["default"]["count"] == 0:
            lines.append(f"⚠️  无人脸检测，建议手动确认或更换图片")
        else:
            lines.append(f"✅ 诊断通过，参考图可以正常使用")
            if crop.get("needs_crop"):
                lines.append(f"   （自动裁切后使用）")

        return "\n".join(lines)

    def save_crop_preview(self, output_path: str = None) -> str:
        """保存裁切预览图到文件"""
        if self.crop_region is None:
            return None

        crop = self.crop_region["crop"]
        face = self.crop_region["face"]
        x, y, fw, fh = face["x"], face["y"], face["fw"], face["fh"]

        # 在原图上画出人脸框和裁切框
        vis = self.img.copy()
        # 人脸框（绿）
        cv2.rectangle(vis, (x, y), (x+fw, y+fh), (0, 255, 0), 3)
        # 裁切框（红）
        cv2.rectangle(vis, (crop["x1"], crop["y1"]), (crop["x2"], crop["y2"]), (0, 0, 255), 3)
        # 人脸中心点（白）
        cv2.circle(vis, (x + fw//2, y + fh//2), 5, (255, 255, 255), -1)

        # 裁切区域
        cropped = self.img[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]

        if output_path is None:
            tmpdir = tempfile.gettempdir()
            output_path = os.path.join(tmpdir, f"crop_preview_{Path(self.img_path).stem}.jpg")

        cv2.imwrite(output_path, vis)
        crop_path = output_path.replace(".jpg", "_crop.jpg")
        cv2.imwrite(crop_path, cropped)

        return output_path, crop_path


def main():
    parser = argparse.ArgumentParser(description="参考图诊断预检工具")
    parser.add_argument("image", help="参考图路径")
    parser.add_argument("--fix", action="store_true", help="自动裁切并保存修复后的参考图到 /tmp")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    diag = ReferenceDiagnostic(args.image)
    if not diag.load():
        print(diag.generate_report())
        sys.exit(1)

    report = diag.generate_report()
    print(report)

    # 保存可视化预览
    try:
        paths = diag.save_crop_preview()
        if paths:
            print(f"\n📊 可视化预览已保存:")
            print(f"   原图标记: {paths[0]}")
            print(f"   裁切结果: {paths[1]}")
    except Exception as e:
        print(f"\n⚠️  预览图保存失败: {e}")

    # 自动修复模式
    if args.fix:
        print(f"\n🔧 自动修复模式:")
        if diag.crop_region and diag.crop_region.get("needs_crop"):
            crop = diag.crop_region["crop"]
            cropped = diag.img[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]
            out_path = os.path.join(tempfile.gettempdir(), f"fixed_ref_{Path(args.image).stem}.jpg")
            cv2.imwrite(out_path, cropped)
            print(f"   ✅ 裁切并保存: {out_path} ({crop['w']}×{crop['h']})")
        else:
            print(f"   ℹ️  无需裁切，直接使用原图")


if __name__ == "__main__":
    main()
