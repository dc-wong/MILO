import cv2
import torch
import numpy as np
from torch import nn
from torchvision import transforms

from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights, ssdlite


# Load SSD300 with VGG16 backbone
model = ssdlite320_mobilenet_v3_large(num_classes = 2)  # Set to True if you want pretrained weights
# model.to(device)

# # Compute the in_channels for each feature map head as before:
# in_channels = [list(m.parameters())[0].shape[0] for m in model.head.classification_head.module_list]

# num_anchors = model.head.classification_head.module_list[0][1].out_channels // 91

# model.head.classification_head = ssdlite.SSDLiteClassificationHead(
#     in_channels=in_channels,
#     num_anchors=[num_anchors],  # now a list for each feature map
#     num_classes=2,
#     norm_layer=nn.BatchNorm2d
# )

# # Modify for 1080p Input
# model.size = (1080, 1920)

# Load the trained model
model_path = "model_default.pth"  # Adjust if necessary
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda'), weights_only=True))
model.to('cuda')
model.eval()


def detect_faces(frame):
    """ Runs the model on the frame and extracts bounding boxes."""
    # image_tensor = preprocess_frame(frame)
    image_tensor = frame
    with torch.no_grad():
        output = model(image_tensor)
    
    # Assuming the model returns bounding boxes in format [x_min, y_min, x_max, y_max]
    bboxes = output["boxes"].cpu().numpy() if "boxes" in output else []
    return bboxes

def draw_boxes(frame, bboxes):
    """ Draws bounding boxes on the frame."""
    for box in bboxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return frame

def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set height


    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # frame = cv2.imread(r"C:\PROJECTS\MILO\MILO\FaceRecognition\data\Dataset_FDDB\aug_images\img_2_top_middle.jpg")

        # frame = cv2.resize(frame, (1920, 1080))
        
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(frame)
        # if len(frame.shape) == 2:
        #     # Grayscale image; convert to 3-channel RGB
        #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        # elif frame.shape[2] == 3:
        #     # Color image assumed to be in BGR; convert to RGB
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        # frame = np.ascontiguousarray(frame, dtype=np.float32)   
        draw_frame = frame.copy()

        # Convert to PyTorch tensor and normalize to [0,1]
        image = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0 
        with torch.no_grad():
            outputs = model([image.to('cuda')])
            # outputs = model([image.to('cuda')])

        boxes = outputs[0]["boxes"]
        scores = outputs[0]["scores"]
        labels = outputs[0]["labels"]

        confidence_threshold = 0.4

        # Draw each bounding box above the threshold
        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                # ---------------------------
                # Step 4: Draw bounding boxes
                # ---------------------------
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw rectangle on the copy of the original BGR frame
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Optionally, put text for label or confidence score
                text = f"ID:{label} | {score:.2f}"
                cv2.putText(draw_frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Display the frame
        cv2.imshow('Webcam', cv2.cvtColor(draw_frame, cv2.COLOR_RGB2BGR))
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
        
    #     bboxes = detect_faces(frame)
    #     frame = draw_boxes(frame, bboxes)
        
    #     cv2.imshow('Webcam Detection', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # cap.release()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
