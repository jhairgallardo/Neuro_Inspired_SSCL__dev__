import torch

def knn_eval(train_loader, val_loader, view_encoder, device, k, num_classes):
    view_encoder.eval()

    ### Get train features and labels
    train_features = []
    train_labels = []
    with torch.no_grad():
        for i, (imgs, labels, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            features = view_encoder(imgs)
            # global average pooling
            features = torch.mean(features, dim=(2,3))
            # flattening
            features = torch.flatten(features, 1)
            train_features.append(features)
            train_labels.append(labels)
    train_features = torch.cat(train_features, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    ### Get val features and labels
    val_features = []
    val_labels = []
    with torch.no_grad():
        for i, (imgs, labels, _) in enumerate(val_loader):
            imgs = imgs.to(device)
            features = view_encoder(imgs)
            features = torch.mean(features, dim=(2,3))
            features = torch.flatten(features, 1)
            val_features.append(features)
            val_labels.append(labels)
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)

    ### KNN
    top1 = knn_classifier(train_features, train_labels, val_features, val_labels, num_classes=num_classes, k=k)

    return top1

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, num_classes, k=20, batch_size=256):
    """
    KNN classification in batches to reduce memory usage.
    """
    num_test = test_features.size(0)
    predictions = []
    for start_idx in range(0, num_test, batch_size):
        end_idx = min(start_idx + batch_size, num_test)
        test_batch = test_features[start_idx:end_idx]  # shape: (B, D)
        distances = torch.cdist(test_batch, train_features, p=2)  # shape: (B, N)
        _, knn_indices = distances.topk(k, dim=1, largest=False)
        nn_labels = train_labels[knn_indices]
        batch_preds = []
        for row in nn_labels:
            counts = row.bincount(minlength=num_classes)
            pred_label = torch.argmax(counts)
            batch_preds.append(pred_label)
        batch_preds = torch.stack(batch_preds)
        predictions.append(batch_preds)
    predictions = torch.cat(predictions, dim=0)  # shape: (num_test,)
    correct = (predictions == test_labels).sum().item()
    accuracy = correct / num_test
    return accuracy


        

    
