import os
import sys
import urllib.request
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# 1. Dynamic Path Resolution (Works for both standalone CLI and Kaggle/HPC environments)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else "/kaggle/working/zero_shot_image_segmentation"
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Suppress Matplotlib font warning popups
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

import configs.pipeline_config
from configs.pipeline_config import config
from scripts.pipeline import IndicZeroShotSegmentationPipeline


def draw_indic_overlay_pil(image: Image.Image, results: list, font_path: str, opacity: float = 0.5, save_path: str = None):
    """
    Overlays colored semantic segmentation masks and burns Indic labels using PIL.
    """
    img_np = np.array(image)
    overlay = img_np.copy()
    
    colors = [
        np.array([255, 60, 60]),    # Coral Red
        np.array([60, 255, 60]),    # Emerald Green
        np.array([60, 160, 255]),   # Azure Blue
        np.array([255, 210, 50]),   # Amber Yellow
    ]

    # Apply alpha blending for masks
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        mask = res["mask"]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask, 
                overlay[:, :, c] * (1 - opacity) + color[c] * opacity, 
                overlay[:, :, c]
            )

    blended_img = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(blended_img)
    
    try:
        font = ImageFont.truetype(font_path, size=22)
    except IOError:
        font = ImageFont.load_default()

    # Draw bounding boxes and text labels
    for i, res in enumerate(results):
        color_tuple = tuple(colors[i % len(colors)].tolist())
        x_min, y_min, x_max, y_max = res["box"]
        indic_lbl = res["indic_label"]
        score = res["final_score"]

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color_tuple, width=4)
        
        # Render clean text string
        label_text = f"{indic_lbl} {score:.2f}"
        bbox_text = draw.textbbox((x_min, max(0, y_min - 32)), label_text, font=font)
        draw.rectangle(bbox_text, fill=color_tuple)
        draw.text((x_min + 4, max(0, y_min - 32)), label_text, fill=(255, 255, 255), font=font)

    # Save output if a path is specified
    if save_path:
        blended_img.save(save_path)
        print(f"Saved segmentation visualization to: {save_path}")

    # Plot figure
    plt.figure(figsize=(14, 9))
    plt.imshow(blended_img)
    plt.axis("off")
    plt.title("Indic-Guided Open-Vocabulary Segmentation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Download Devanagari TrueType font if not present
    font_path = os.path.join(current_dir, "NotoDevanagari.ttf")
    if not os.path.exists(font_path):
        font_url = "https://raw.githubusercontent.com/googlefonts/noto-fonts/main/unhinted/ttf/NotoSansDevanagari/NotoSansDevanagari-Regular.ttf"
        urllib.request.urlretrieve(font_url, font_path)

    # 2. Initialize the Multimodal Pipeline
    pipeline = IndicZeroShotSegmentationPipeline(config)

    # 3. Load Sample Test Image
    img_url = "https://images.unsplash.com/photo-1581092160607-ee22621dd758?q=80&w=1000&auto=format&fit=crop"
    test_img_path = os.path.join(current_dir, "test_industrial.jpg")
    if not os.path.exists(test_img_path):
        urllib.request.urlretrieve(img_url, test_img_path)
    test_image = Image.open(test_img_path).convert("RGB")

    # 4. Define Regional Prompts
    hindi_prompts = [
        "सुरक्षा चश्मा",       # safety glasses
        "नीली जैकेट",         # blue jacket
        "धातु का पाइप",       # metal pipe
        "पीला हेलमेट"         # yellow helmet (negative test)
    ]

    # 5. Run Prediction
    predictions = pipeline.predict(
        image=test_image,
        indic_queries=hindi_prompts,
        source_lang="hin_Deva"
    )

    print(f"\nTotal verified masks extracted: {len(predictions)}")
    for idx, pred in enumerate(predictions, 1):
        print(f"  [{idx}] '{pred['indic_label']}' ({pred['english_label']}): S_final = {pred['final_score']:.3f}")

    # 6. Render Output
    if predictions:
        output_result_path = os.path.join(current_dir, "output_segmented.png")
        draw_indic_overlay_pil(
            test_image, 
            predictions, 
            font_path=font_path, 
            opacity=config.mask_opacity,
            save_path=output_result_path
        )


if __name__ == "__main__":
    main()
