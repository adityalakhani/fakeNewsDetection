import torch
from transformers import AutoModel, BertTokenizerFast
import torch.nn as nn

# Load BERT model and tokenizer
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Load trained model
class BERT_Arch(torch.nn.Module):
    def __init__(self, bert):  
        super(BERT_Arch, self).__init__()
        self.bert = bert   
        self.dropout = nn.Dropout(0.1)            # dropout layer
        self.relu = nn.ReLU()                      # relu activation function
        self.fc1 = nn.Linear(768, 512)             # dense layer 1
        self.fc2 = nn.Linear(512, 2)               # dense layer 2 (Output layer)
        self.softmax = nn.LogSoftmax(dim=1)        # softmax activation function
    
    def forward(self, sent_id, mask):              # define the forward pass  
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)                           # output layer
        x = self.softmax(x)                       # apply softmax activation
        return x

model = BERT_Arch(bert)
model_path = 'c1_fakenews_weights.pt'
model_state_dict = torch.load(model_path)
unexpected_keys = [k for k in model_state_dict.keys() if 'position_ids' in k]
for k in unexpected_keys:
    del model_state_dict[k]
model.load_state_dict(model_state_dict)
model.eval()

def prediction(news_headline):
    # news_headline = request.form['news_headline']
    # Tokenize and encode input
    tokens = tokenizer.encode_plus(news_headline, max_length=15, pad_to_max_length=True, truncation=True)
    input_ids = torch.tensor(tokens['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokens['attention_mask']).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, axis=1).item()

    prediction = "Fake" if preds == 0 else "True"
    return prediction