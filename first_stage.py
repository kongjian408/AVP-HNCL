import os
import joblib
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from util.seed import set_seed
from util.data import load_data
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.utils import class_weight
from sklearn.preprocessing import StandardScaler
from util.augmentation import augment_sequence
from util.Queue import Queue
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score, f1_score

set_seed()

torch.cuda.empty_cache()

train_file = './dataset/Set 1/train.txt'
test_file = './dataset/Set 1/test.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

esm2 = AutoModel.from_pretrained('./esm2').to(device)
tokenizer = AutoTokenizer.from_pretrained('./esm2')
esm2.eval()

pos_train, neg_train = load_data(train_file)
pos_test, neg_test = load_data(test_file)

y_train = np.array([1] * len(pos_train) + [0] * len(neg_train))
y_test = np.array([1] * len(pos_test) + [0] * len(neg_test))

train_seq = pos_train + neg_train
test_seq = pos_test + neg_test

from util.data import generate_features

train_feat = generate_features(train_file)
test_feat = generate_features(test_file)

train_feat_clipped = np.clip(train_feat, -10, 10)
test_feat_clipped = np.clip(test_feat, -10, 10)

train_feat_log_sigmoid = np.log1p(np.exp(train_feat_clipped))
test_feat_log_sigmoid = np.log1p(np.exp(test_feat_clipped))

scaler = StandardScaler()
additional_train_features = pd.DataFrame(
    scaler.fit_transform(train_feat_log_sigmoid),
    columns=train_feat.columns
)
additional_test_features = pd.DataFrame(
    scaler.transform(test_feat_log_sigmoid),
    columns=test_feat.columns
)

max_length = 45

from util.data import esm_encode

X_train = esm_encode(
    train_seq,
    esm2,
    tokenizer,
    device,
    max_length=max_length,
    additional_features=additional_train_features.values
)

X_test = esm_encode(
    test_seq,
    esm2,
    tokenizer,
    device,
    max_length=max_length,
    additional_features=additional_test_features.values
)

X_train_tensor = X_train.clone().detach().to(device)
X_test_tensor = X_test.clone().detach().to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

from loss import ContrastiveLoss

hard_neg_contra_loss = ContrastiveLoss(
    temperature=0.5,
    margin=0.2,
    learnable_temperature=True,
    regularization=1e-4
)

from model import FeatureExtractor,Classifier
from model2 import ModelCombiner

input_dim = 1071
embedding_dim = 505
hidden_dim = 960
lstm_hidden_dim = 512
num_classes = 2
dropout_rate = 0.42

contra_weight = 0.1
epochs = 50
batch_size = 32
k = 10
num_fragments = 6
multi_step = 1
queue_size = 3000

feature_extractor = FeatureExtractor(input_dim, embedding_dim, hidden_dim, lstm_hidden_dim)
classifier = Classifier(lstm_hidden_dim, num_classes, dropout_rate)
model = ModelCombiner(feature_extractor, classifier).to(device)

def train():
    class_weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    loss_history, acc_history = [], []


    embedding_dim = X_train_tensor.shape[1]
    pos_queue = Queue(max_size=queue_size, embedding_dim=embedding_dim, device=device)
    neg_queue = Queue(max_size=queue_size, embedding_dim=embedding_dim, device=device)
    final_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        permutation = torch.randperm(X_train_tensor.size(0))
        X_train_shuffled = X_train_tensor[permutation]
        y_train_shuffled = y_train_tensor[permutation]
        add_train_shuffled = additional_train_features.iloc[permutation.cpu().numpy()].values
        train_sequences_shuffled = [train_seq[idx] for idx in permutation.cpu().numpy()]

        for i in tqdm(range(0, X_train_tensor.size(0), batch_size), desc=f'Epoch {epoch + 1}/{epochs}', unit='Batch'):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]
            batch_additional = add_train_shuffled[i:i + batch_size]
            batch_sequences = train_sequences_shuffled[i:i + batch_size]

            aug_sequences = [augment_sequence(seq,num_fragments=num_fragments,multi_step=multi_step) for seq in batch_sequences]
            aug_X = esm_encode(
                aug_sequences,
                esm2,
                tokenizer,
                device,
                max_length=max_length,
                additional_features=batch_additional
            )
            aug_X_tensor = aug_X.clone().detach().to(device)

            optimizer.zero_grad()
            outputs, embeddings = model(batch_X)
            _, aug_embeddings = model(
                aug_X_tensor)

            classification_loss = criterion(outputs, batch_y)

            pos_indices = (batch_y == 1)
            neg_indices = (batch_y == 0)

            if pos_indices.any():
                pos_embeddings = embeddings[pos_indices]
                pos_queue.enqueue(pos_embeddings)
            if neg_indices.any():
                neg_embeddings_batch = embeddings[neg_indices]
                neg_queue.enqueue(neg_embeddings_batch)

            hard_neg_list = []

            neg_embeddings = neg_queue.get_all_embeddings()
            pos_embeddings_store = pos_queue.get_all_embeddings()

            if pos_indices.any() and neg_embeddings.size(0) > 0:
                pos_embeddings = embeddings[pos_indices]
                similarity_pos_neg = F.cosine_similarity(
                    pos_embeddings.unsqueeze(1), neg_embeddings.unsqueeze(0), dim=2
                )

                topk_values_pos, topk_indices_pos = similarity_pos_neg.topk(k, dim=1, largest=True,
                                                                            sorted=True)
                hard_neg_pos = neg_embeddings[topk_indices_pos]
                hard_neg_list.append(hard_neg_pos)

            if neg_indices.any() and pos_embeddings_store.size(0) > 0:
                neg_embeddings = embeddings[neg_indices]
                similarity_neg_pos = F.cosine_similarity(
                    neg_embeddings.unsqueeze(1), pos_embeddings_store.unsqueeze(0), dim=2
                )

                topk_values_neg, topk_indices_neg = similarity_neg_pos.topk(k, dim=1, largest=True,
                                                                            sorted=True)
                hard_neg_neg = pos_embeddings_store[topk_indices_neg]
                hard_neg_list.append(hard_neg_neg)

            if hard_neg_list:
                hard_neg = torch.cat(hard_neg_list, dim=0)
            else:
                hard_neg = torch.zeros(0, k, embedding_dim).to(device)

            if hard_neg.size(0) > 0:
                con_loss = hard_neg_contra_loss(
                    torch.cat([embeddings[pos_indices], embeddings[neg_indices]], dim=0),
                    torch.cat([aug_embeddings[pos_indices], aug_embeddings[neg_indices]], dim=0),
                    hard_neg
                )
                loss = classification_loss + contra_weight * con_loss
            else:
                loss = classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / X_train_tensor.size(0)
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)

        print(
            f'Epoch {epoch + 1}, Train loss: {epoch_loss:.4f}, Train ACC: {epoch_acc:.4f}')

        scheduler.step()

    torch.save(feature_extractor.state_dict(), './dataset/Set 1/parm/feature_extractor.pth')
    torch.save(model.state_dict(), './dataset/Set 1/parm/model.pth')

def evaluate():
    model.load_state_dict(torch.load('./dataset/Set 1/parm/model.pth'))
    model.eval()

    class_weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    test_running_loss, test_correct, test_total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        test_outputs, test_embeddings = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        test_running_loss += test_loss.item() * X_test_tensor.size(0)

        _, test_predicted = torch.max(test_outputs, 1)
        test_total += y_test_tensor.size(0)
        test_correct += (test_predicted == y_test_tensor).sum().item()

        all_preds.extend(test_predicted.cpu().numpy())
        all_labels.extend(y_test_tensor.cpu().numpy())

        softmax_probs = F.softmax(test_outputs, dim=1)
        all_probs.extend(softmax_probs.cpu().numpy()[:, 1])

    test_loss = test_running_loss / X_test_tensor.size(0)
    test_acc = test_correct / test_total

    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()

    SN = TP / (TP + FN) if (TP + FN) > 0 else 0
    SP = TN / (TN + FP) if (TN + FP) > 0 else 0

    MCC = matthews_corrcoef(all_labels, all_preds)
    GMean = (SN * SP) ** 0.5

    AUPRC = average_precision_score(all_labels, all_probs)
    AUROC = roc_auc_score(all_labels, all_probs)
    F1 = f1_score(all_labels, all_preds)

    print(f'Test Loss: {test_loss:.4f}, Test ACC: {test_acc:.4f}')
    print(f'MCC: {MCC:.4f}')
    print(f'SN: {SN:.4f}')
    print(f'SP: {SP:.4f}')
    print(f'G-Mean: {GMean:.4f}')
    print(f'AUPRC: {AUPRC:.4f}')
    print(f'AUROC: {AUROC:.4f}')
    print(f'F1-Score: {F1:.4f}')
    print(f'CM: \n{cm}')



if __name__ == '__main__':
    evaluate()
