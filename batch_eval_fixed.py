import os, sys, glob
import tensorflow as tf
import numpy as np
from tqdm import tqdm

LABELS = 'logs/output_labels.txt'
GRAPH = 'logs/output_graph.pb'

def load_graph(graph_file):
    with tf.io.gfile.GFile(graph_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

def read_labels(label_file):
    with open(label_file, 'r') as f:
        return [l.strip() for l in f.readlines()]

def run_inference_on_image(sess, softmax_tensor, image_data):
    preds = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    return np.squeeze(preds)

def evaluate(root_dir):
    labels = read_labels(LABELS)                       # labels like 'a','b',...
    labels_lc = [l.lower() for l in labels]
    graph = load_graph(GRAPH)
    softmax_tensor = graph.get_tensor_by_name('final_result:0')
    total = 0
    correct = 0
    per_class = {}
    with tf.compat.v1.Session(graph=graph) as sess:
        for class_dir in sorted(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if not os.path.isdir(class_path): continue
            for img in tqdm(glob.glob(os.path.join(class_path, '*')), desc=class_dir):
                try:
                    with open(img, 'rb') as f:
                        image_data = f.read()
                except:
                    continue
                preds = run_inference_on_image(sess, softmax_tensor, image_data)
                top = preds.argmax()
                pred_label = labels[top]
                total += 1
                per_class.setdefault(class_dir, {'total':0,'correct':0})
                per_class[class_dir]['total'] += 1
                # compare case-insensitively
                if pred_label.lower() == class_dir.lower():
                    per_class[class_dir]['correct'] += 1
                    correct += 1

    print("\n==== FINAL RESULTS ====")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {100.0*correct/total:.2f}%\n")
    for k in sorted(per_class.keys(), key=lambda x: x.lower()):
        d = per_class[k]
        print(f"{k:5s}: {d['correct']:4d} / {d['total']:4d}  → {100.0*d['correct']/d['total']:.2f}%")

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv)>1 else 'dataset_subset'
    evaluate(root)
