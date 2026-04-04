import matplotlib.pyplot as plt
import cv2

def plot_prediction(image, bboxes, labels, scores, output_path):
    """
    在图像上绘制RAViT选中的Top-k区域及分类标签
    """
    img = cv2.imread(image)
    for box, label, score in zip(bboxes, labels, scores):
        if score > 0.3: # 置信度阈值
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(img, f"{label}: {score:.2f}", (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imwrite(output_path, img)