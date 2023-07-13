import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

json_files = [
    os.path.join("result", "data_april14_Celeb-DF.json"),
    os.path.join("result", "data_april14_DFDC.json"),
    os.path.join("result", "data_april11_DeepfakeTIMIT.json"),
    os.path.join("result", "data_april14_FF++.json"),
]

# Lists to store the ROC curve data
fpr_list = []
tpr_list = []
roc_auc_list = []

for json_file in json_files:
    with open(json_file, "r") as f:
        result = json.load(f)

    # Get the actual labels and predicted probabilities or predicted labels from the result dictionary
    actual_labels = result["video"]["correct_label"]
    predicted_probs = result["video"]["pred"]
    predicted_labels = result["video"]["pred_label"]

    big_pp = [1 if P >= 0.5 else 0 for P in predicted_probs]
    p_labels = [1 if label == "FAKE" else 0 for label in predicted_labels]
    a_labels = [1 if label == "FAKE" else 0 for label in actual_labels]

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(a_labels, predicted_probs)
    roc_auc = roc_auc_score(a_labels, predicted_probs)
    f1 = f1_score(a_labels, big_pp)

    # Append the data to the lists
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    roc_auc_list.append(roc_auc)

    a = 0
    for i in range(len(p_labels)):
        if p_labels[i] == a_labels[i]:
            a += 1

    accuracy = sum(x == y for x, y in zip(p_labels, a_labels)) / len(p_labels)
    real_acc = sum(
        (x == y and y == 0) for x, y in zip(p_labels, a_labels)
    ) / a_labels.count(0)
    fake_acc = sum(
        (x == y and y == 1) for x, y in zip(p_labels, a_labels)
    ) / a_labels.count(1)
    print(
        f"{(json_file[:-5].split('_')[-1])}:\nReal accuracy {real_acc*100:.3f} Fake accuracy {fake_acc*100:.3f}, Accuracy: {accuracy*100:.3f}"
    )
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"F1 Score: {f1:.3f}\n")

# Plot ROC curves
plt.figure()
for i in range(len(json_files)):
    plt.plot(
        fpr_list[i],
        tpr_list[i],
        label=f"{json_files[i][:-5].split('_')[-1]} (area = %0.3f)" % roc_auc_list[i],
    )

plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()
