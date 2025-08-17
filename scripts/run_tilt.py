import argparse
from tilt_worker.src.pipelines.tilt_pipeline import process_video

def main():
    ap = argparse.ArgumentParser(description="Tilt Worker Runner")
    ap.add_argument("--video", required=True, help="분석할 비디오 파일 경로")
    ap.add_argument("--threshold", type=float, default=3.0, help="Tilt threshold (deg)")
    ap.add_argument("--cross-scale", type=float, default=0.6, help="어깨폭 대비 십자가 반길이 배수")
    ap.add_argument("--cross-min", type=int, default=30, help="십자가 최소 반길이(px)")
    ap.add_argument("--cross-max-ratio", type=float, default=0.4, help="화면 높이 대비 십자가 반길이 최대 비율")
    ap.add_argument("--thick-mul", type=float, default=0.06, help="어깨폭 대비 선 두께 배수")
    ap.add_argument("--font-mul", type=float, default=0.025, help="어깨폭 대비 폰트 스케일 배수")
    ap.add_argument("--no-median", action="store_true", help="중앙값 필터 비활성화")
    ap.add_argument("--no-savgol", action="store_true", help="Savgol 필터 비활성화")

    args = ap.parse_args()

    result = process_video(
        video_path=args.video,
        threshold_deg=args.threshold,
        do_median=not args.no_median,
        do_savgol=not args.no_savgol,
        cross_scale=args.cross_scale,
        cross_min=args.cross_min,
        cross_max_ratio=args.cross_max_ratio,
        thick_mul=args.thick_mul,
        font_mul=args.font_mul
    )

    print("[DONE] Tilt analysis finished.")
    print(result)


if __name__ == "__main__":
    main()
