import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO  # adjust import as needed
from helping_functions import helper
config_file = 'Your config file here'
try:
    # Read the configuration values from config             
    configuration_values = helper().read_config(config_file)
    #print(configuration_values)
except Exception as ex:
    print('Reading Configaration Error :', ex)



class TensorflowDetection:
    def __init__(self, model_path):
        self.detection_graph = tf.Graph()
        self.helper = helper()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def process(self, image_np, draw_boxes=False, threshold=0.5):
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_exp = np.expand_dims(image_np, axis=0)
        boxes, scores, classes, num = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_exp}
        )
        class_ids, det_scores, bboxes, count = self.helper.filterBB(
            classes, scores, boxes, threshold=threshold
        )
        if draw_boxes:
            h, w, _ = image_bgr.shape
            for cid, score, box in zip(class_ids, det_scores, bboxes):
                y1, x1, y2, x2 = box
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(image_bgr, p1, p2, (0, 255, 0), 2)
                label = f"{cid}:{score:.2f}"
                cv2.putText(image_bgr, label, (p1[0], p1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return image_bgr, bboxes, class_ids, det_scores, count
        return bboxes, class_ids, det_scores, count


class YoloDetection:
    def __init__(self, model_path):
        self.yolo = YOLO(model_path, 'segment')

    def process(self, image_np, draw_boxes=False, conf=0.7):
        h, w, _ = image_np.shape
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = self.yolo.predict(
            source=image_bgr, verbose=False, conf=conf,
            line_width=2, retina_masks=False, save=False, save_conf=False
        )
        boxes = results[0].boxes
        bboxes, class_ids, det_scores = [], [], []
        for i, box in enumerate(boxes.xyxy):
            x1, y1, x2, y2 = box.tolist()
            score = float(boxes.conf[i])
            cid = int(boxes.cls[i]) + 1
            norm_box = [y1/h, x1/w, y2/h, x2/w]
            bboxes.append(norm_box)
            class_ids.append(cid)
            det_scores.append(score)
        count = len(bboxes)
        if draw_boxes:
            for norm_box, cid, score in zip(bboxes, class_ids, det_scores):
                y1, x1, y2, x2 = norm_box
                p1 = (int(x1 * w), int(y1 * h))
                p2 = (int(x2 * w), int(y2 * h))
                cv2.rectangle(image_bgr, p1, p2, (255, 0, 0), 2)
                label = f"{cid}:{score:.2f}"
                cv2.putText(image_bgr, label, (p1[0], p1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            return image_bgr, bboxes, class_ids, det_scores, count
        return bboxes, class_ids, det_scores, count


def detect_image(model_name: str, model_path: str, image: np.ndarray, draw: bool):
    """
    model_name: 'tf' or 'yolo'
    model_path: path to the model file
    image: input image as a NumPy array (RGB)
    draw: True to draw boxes, False for coords only
    """
    if model_name.lower() == 'tf':
        detector = TensorflowDetection(model_path)
    elif model_name.lower() == 'yolo':
        detector = YoloDetection(model_path)
    else:
        raise ValueError("model_name must be 'tf' or 'yolo'")

    return detector.process(image, draw_boxes=draw)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run object detection")
    parser.add_argument("--model", choices=["tf", "yolo"], required=True,
                        help="Which model to use: 'tf' or 'yolo'")
    parser.add_argument("--model_path", required=True, help="Path to the model file")
    parser.add_argument("--image", required=True,
                        help="Path to input image")
    parser.add_argument("--draw", action="store_true",
                        help="Draw bounding boxes on the image")
    args = parser.parse_args()

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    output = detect_image(args.model, args.model_path, img, args.draw)

    # If drawing, output is (image, boxes, classes, scores, count)
    # Otherwise (boxes, classes, scores, count)
    if args.draw:
        result_img, boxes, classes, scores, count = output
        cv2.imshow("Detections", result_img)
        cv2.waitKey(0)
    else:
        boxes, classes, scores, count = output
        print("Boxes:", boxes)
        print("Classes:", classes)
        print("Scores:", scores)
        print("Count:", count)
