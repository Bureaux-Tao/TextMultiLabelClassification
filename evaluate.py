from loader import load_data
from path import test_file_path
from predict import mlb, predict
from tqdm import tqdm
from sklearn.metrics import classification_report


def evaluate():
    test_x, test_y = load_data(test_file_path)
    true_y_list = mlb.transform(test_y)
    
    pred_y_list = []
    for text in tqdm(test_x):
        pred_y, pred_label = predict(text)
        pred_y_list.append(pred_y)
    # F1值
    print(classification_report(true_y_list, pred_y_list, digits = 4, target_names = mlb.classes_.tolist()))


if __name__ == '__main__':
    evaluate()