import torch

# モデルの評価
def evaluate_model(dataloader, tokenizer, model, device):
    model.eval()
    total_predictions = []

    # 推論時には基本torch.no_grad()の中でする
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            _, predictions = torch.max(outputs.logits, dim=1)

            total_predictions.extend(predictions.detach().cpu().numpy())